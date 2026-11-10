"""Noise models for hyperspectral denoising, batched on the GPU.

Every function takes clean cubes ``x`` of shape ``(batch, bands, rows, columns)`` in ``[0, 1]`` and a
``torch.Generator`` on the same device, and returns a noisy copy. ``rows`` is the along-track axis and
``columns`` the cross-track detector axis of a pushbroom sensor, so detector faults are constant down a column.

Standard noise reproduces the MP-HSIR training and test degradations (Gaussian, non-i.i.d. Gaussian, stripe,
deadline, impulse). Physics noise is direction D1 of the plan: signal-dependent Poisson-Gaussian noise, per-detector
gain/offset stripes, a band-dependent SNR profile and dead/saturated pixels. Its parameter ranges come from the
literature, not from fitted L1 products, because no L1 scenes are available to this pilot.
"""

import math

import torch

NON_IID_SIGMAS = (10, 30, 50, 70)
IMPULSE_AMOUNTS = (0.1, 0.3, 0.5, 0.7)
TARGET_WAVELENGTHS = torch.linspace(400.0, 1000.0, 100)


def _uniform(shape, low, high, generator, device):
    return low + (high - low) * torch.rand(shape, generator=generator, device=device)


def _log_uniform(shape, low, high, generator, device):
    return torch.exp(_uniform(shape, math.log(low), math.log(high), generator, device))


def gaussian(x, sigma_low, sigma_high, generator):
    """i.i.d. Gaussian, one sigma per cube drawn from ``U(sigma_low, sigma_high) / 255``."""
    sigma = (
        _uniform((x.shape[0], 1, 1, 1), sigma_low, sigma_high, generator, x.device)
        / 255.0
    )
    return x + sigma * torch.randn(x.shape, generator=generator, device=x.device)


def gaussian_fixed(x, sigma, generator):
    return x + (sigma / 255.0) * torch.randn(
        x.shape, generator=generator, device=x.device
    )


def non_iid(x, generator):
    """Band-wise sigma drawn from {10, 30, 50, 70} / 255 (MP-HSIR complex-noise base)."""
    sigmas = torch.tensor(NON_IID_SIGMAS, device=x.device, dtype=x.dtype) / 255.0
    index = torch.randint(
        0,
        len(NON_IID_SIGMAS),
        (x.shape[0], x.shape[1], 1, 1),
        generator=generator,
        device=x.device,
    )
    return x + sigmas[index] * torch.randn(
        x.shape, generator=generator, device=x.device
    )


def _band_column_mask(x, band_fraction, low, high, generator):
    """Boolean ``(batch, bands, 1, columns)``: a third of the bands get ``U(low, high)`` of their columns."""
    batch, bands, _, columns = x.shape
    affected = max(1, int(band_fraction * bands))
    band_scores = torch.rand((batch, bands), generator=generator, device=x.device)
    band_rank = band_scores.argsort(dim=1).argsort(dim=1)
    band_selected = band_rank < affected
    count = torch.randint(
        int(low * columns),
        max(int(high * columns), int(low * columns) + 1),
        (batch, bands, 1),
        generator=generator,
        device=x.device,
    )
    column_rank = (
        torch.rand((batch, bands, columns), generator=generator, device=x.device)
        .argsort(-1)
        .argsort(-1)
    )
    column_selected = column_rank < count
    return (band_selected.unsqueeze(-1) & column_selected).unsqueeze(2)


def stripe(x, generator, low=0.05, high=0.15):
    mask = _band_column_mask(x, 1 / 3, low, high, generator)
    shift = _uniform(mask.shape, -0.25, 0.25, generator, x.device)
    return x - mask * shift


def deadline(x, generator, low=0.05, high=0.15):
    mask = _band_column_mask(x, 1 / 3, low, high, generator)
    return x.masked_fill(mask, 0.0)


def impulse(x, generator, amounts=IMPULSE_AMOUNTS):
    batch, bands = x.shape[:2]
    affected = max(1, int(bands / 3))
    band_rank = (
        torch.rand((batch, bands), generator=generator, device=x.device)
        .argsort(1)
        .argsort(1)
    )
    band_selected = (band_rank < affected)[:, :, None, None]
    choice = torch.randint(
        0, len(amounts), (batch, 1, 1, 1), generator=generator, device=x.device
    )
    amount = torch.tensor(amounts, device=x.device, dtype=x.dtype)[choice]
    flipped = (
        torch.rand(x.shape, generator=generator, device=x.device) < amount
    ) & band_selected
    salted = torch.rand(x.shape, generator=generator, device=x.device) < 0.5
    out = x.masked_fill(flipped & salted, 1.0)
    return out.masked_fill(flipped & ~salted, 0.0)


def complex_noise(x, generator):
    """MP-HSIR ``complexN`` training mix: non-i.i.d. Gaussian plus one of deadline, impulse or stripe per cube."""
    noisy = non_iid(x, generator)
    kind = torch.randint(0, 3, (x.shape[0],), generator=generator, device=x.device)
    out = noisy.clone()
    for k, fn in enumerate((deadline, impulse, stripe)):
        selected = kind == k
        if selected.any():
            out[selected] = fn(noisy[selected], generator)
    return out


def standard(x, generator):
    """Half Gaussian (prompt 0), half complex (prompt 1). Returns ``(noisy, prompt)``."""
    is_complex = torch.rand((x.shape[0],), generator=generator, device=x.device) < 0.5
    out = gaussian(x, 30, 70, generator)
    if is_complex.any():
        out[is_complex] = complex_noise(x[is_complex], generator)
    return out, is_complex.long()


def snr_profile(bands, generator, device, wavelengths=None):
    """Noise multiplier per band, heavier at the blue and NIR edges where silicon detectors lose sensitivity."""
    wl = (wavelengths if wavelengths is not None else TARGET_WAVELENGTHS).to(device)[
        :bands
    ]
    blue = _uniform((1, 1), 0.5, 3.0, generator, device)
    nir = _uniform((1, 1), 0.5, 3.0, generator, device)
    return (
        1.0
        + blue * torch.exp(-(wl - 400.0) / 40.0)
        + nir * torch.exp((wl - 1000.0) / 60.0)
    )


def physics(x, generator, scale=1.0):
    """Pushbroom sensor model (D1). ``scale`` > 1 stresses the noise beyond the training range.

    y = g[b, col] * x + o[b, col] + sqrt(s_b^2 * (a * x + c)) * n, then dead/saturated pixels.
    a ~ logU(1e-4, 2e-2), c ~ logU(1e-5, 4e-3) per cube; s_b is the SNR profile; detector gain/offset
    are partly shared across bands (correlation rho ~ U(0, 0.8)) because one detector element serves all bands.
    """
    batch, bands, _, columns = x.shape
    device = x.device
    a = _log_uniform((batch, 1, 1, 1), 1e-4, 2e-2, generator, device) * scale
    c = _log_uniform((batch, 1, 1, 1), 1e-5, 4e-3, generator, device) * scale
    profile = torch.stack(
        [snr_profile(bands, generator, device) for _ in range(batch)]
    ).view(batch, bands, 1, 1)
    sigma = profile * torch.sqrt(a * x.clamp(min=0) + c)
    noisy = x + sigma * torch.randn(x.shape, generator=generator, device=device)

    rho = _uniform((batch, 1, 1, 1), 0.0, 0.8, generator, device)
    shared = torch.randn((batch, 1, 1, columns), generator=generator, device=device)
    own = torch.randn((batch, bands, 1, columns), generator=generator, device=device)
    gain_std = _uniform((batch, 1, 1, 1), 0.01, 0.05, generator, device) * scale
    offset_std = _uniform((batch, 1, 1, 1), 0.0, 0.02, generator, device) * scale
    gain = 1.0 + gain_std * (rho * shared + torch.sqrt(1 - rho**2) * own)
    offset_noise = torch.randn(
        (batch, bands, 1, columns), generator=generator, device=device
    )
    offset = offset_std * (rho * shared + torch.sqrt(1 - rho**2) * offset_noise)
    noisy = gain * noisy + offset

    dead_fraction = _uniform((batch, 1, 1, 1), 1e-3, 1e-2, generator, device) * scale
    hit = torch.rand(x.shape, generator=generator, device=device) < dead_fraction
    saturated = torch.rand(x.shape, generator=generator, device=device) < 0.3
    noisy = noisy.masked_fill(hit & ~saturated, 0.0).masked_fill(hit & saturated, 1.0)

    dead_columns = (
        torch.rand((batch, 1, 1, 1), generator=generator, device=device) < 0.3
    )
    if dead_columns.any():
        noisy = torch.where(
            dead_columns, deadline(noisy, generator, 0.005, 0.02), noisy
        )
    return noisy


def physics_mix(x, generator, physics_fraction=0.6):
    """D1 arm: ``physics_fraction`` physics noise (prompt 1), the rest standard noise."""
    out, prompt = standard(x, generator)
    use_physics = (
        torch.rand((x.shape[0],), generator=generator, device=x.device)
        < physics_fraction
    )
    if use_physics.any():
        out[use_physics] = physics(x[use_physics], generator)
        prompt[use_physics] = 1
    return out, prompt


EVAL_CONDITIONS = {
    "gauss30": (0, lambda x, g: gaussian_fixed(x, 30, g)),
    "gauss50": (0, lambda x, g: gaussian_fixed(x, 50, g)),
    "gauss70": (0, lambda x, g: gaussian_fixed(x, 70, g)),
    "case1_noniid": (1, non_iid),
    "case2_stripe": (1, lambda x, g: stripe(non_iid(x, g), g)),
    "case3_deadline": (1, lambda x, g: deadline(non_iid(x, g), g)),
    "case4_impulse": (1, lambda x, g: impulse(non_iid(x, g), g)),
    "case4_mixture": (
        1,
        lambda x, g: impulse(deadline(stripe(non_iid(x, g), g), g), g),
    ),
    "physics": (1, physics),
    "physics_heavy": (1, lambda x, g: physics(x, g, scale=2.0)),
}
