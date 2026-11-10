"""Download and prepare remote-sensing HSI data for the MP-HSIR pilot.

Outputs under ``--root``:
  test/<Set>/*.mat        official MP-HSIR test crops (Google Drive), 100 bands, min-max per crop
  train_patches.npy       float16 (N, 100, 64, 64) clean patches from scene regions outside every test crop
  train_sources.json      patch count per source region
  real/<scene>.npy        float32 (100, H, W) HyRank Hyperion scenes for no-reference evaluation

Training regions follow MP-HSIR ``utils/mat_data.py`` so no pixel of a test crop is used for training. Every scene
is resampled to 100 bands on 400-1000 nm with the same linear wavelength model as ``utils/lmdb_patch.py``.
Xiong'an has no public mirror here, so it is a held-out-scene test only.

    python hsi_mphsir/data.py --root /workspace/hsi_data
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import interpolate

TEST_FOLDER = "https://drive.google.com/drive/folders/1EcX6AY63sylH00ixqcD99mAPupDK8KfB"
TEST_SETS = ("WDC", "Xiongan", "Chikusei", "PaviaC", "PaviaU", "Houston")
WAVELENGTH_RANGES = {
    "WDC": (400, 2400, 191),
    "PaviaC": (430, 860, 102),
    "PaviaU": (430, 860, 103),
    "Houston": (364, 1046, 144),
    "Chikusei": (343, 1018, 128),
    "HyRank": (426, 925, 50),
}
TARGET = np.linspace(400, 1000, 100)
PATCH, STRIDE, MAX_PER_REGION = 64, 32, 1500


def to_100_bands(cube, name):
    """``cube`` is (bands, rows, columns); returns float32 (100, rows, columns)."""
    low, high, bands = WAVELENGTH_RANGES[name]
    assert cube.shape[0] == bands, (name, cube.shape)
    source = np.linspace(low, high, bands)
    resample = interpolate.interp1d(
        source, cube.astype(np.float32), axis=0, kind="linear", fill_value="extrapolate"
    )
    return resample(TARGET).astype(np.float32)


def minmax(cube, low_pct=None, high_pct=None):
    low = np.percentile(cube, low_pct) if low_pct is not None else cube.min()
    high = np.percentile(cube, high_pct) if high_pct is not None else cube.max()
    return np.clip((cube - low) / max(high - low, 1e-8), 0, 1).astype(np.float32)


def patches(region, rng):
    _, rows, columns = region.shape
    corners = [
        (r, c)
        for r in range(0, rows - PATCH + 1, STRIDE)
        for c in range(0, columns - PATCH + 1, STRIDE)
    ]
    if len(corners) > MAX_PER_REGION:
        corners = [
            corners[i] for i in rng.choice(len(corners), MAX_PER_REGION, replace=False)
        ]
    return np.stack(
        [region[:, r : r + PATCH, c : c + PATCH] for r, c in corners]
    ).astype(np.float16)


def hf_file(repo, filename):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo, filename, repo_type="dataset", token=os.environ.get("HF_TOKEN")
    )


def largest_array(mat):
    arrays = {k: v for k, v in mat.items() if isinstance(v, np.ndarray) and v.ndim == 3}
    return arrays[max(arrays, key=lambda k: arrays[k].size)]


def training_regions():
    """Yield ``(name, test_crop, regions)``; arrays are (bands, rows, columns) in native units.

    ``test_crop`` is where MP-HSIR cut its official test crop (checked against the downloaded file before use);
    ``regions`` are the training regions outside it.
    """
    import tifffile
    from scipy.io import loadmat

    wdc = tifffile.imread(hf_file("danaroth/washington_dc_mall", "dc.tif")).astype(
        np.float32
    )
    yield "WDC", wdc[:, 510:766, 25:281], [wdc[:, :510, :], wdc[:, 766:, :]]

    pavia_c = (
        loadmat(hf_file("danaroth/pavia", "Pavia.mat"))["pavia"]
        .astype(np.float32)
        .transpose(2, 0, 1)
    )
    yield (
        "PaviaC",
        pavia_c[:, 420:676, 256:512],
        [
            pavia_c[:, :, :230],
            pavia_c[:, :420, 230:],
            pavia_c[:, 420:676, 512:],
            pavia_c[:, 676:, 230:],
        ],
    )

    pavia_u = (
        loadmat(hf_file("danaroth/pavia", "PaviaU.mat"))["paviaU"]
        .astype(np.float32)
        .transpose(2, 0, 1)
    )
    yield "PaviaU", pavia_u[:, 200:400, :], [pavia_u[:, :200, :], pavia_u[:, 400:, :]]

    houston = largest_array(
        loadmat(hf_file("danaroth/grss_dfc_2013", "dfc2013.mat"))
    ).astype(np.float32)
    houston = houston.transpose(2, 0, 1) if houston.shape[-1] == 144 else houston
    yield (
        "Houston",
        houston[:, :, 1024:1280],
        [houston[:, :, :1024], houston[:, :, 1280:]],
    )

    hf_file("danaroth/chikusei", "HyperspecVNIR_Chikusei_20140729.hdr")
    raw = np.fromfile(
        hf_file("danaroth/chikusei", "HyperspecVNIR_Chikusei_20140729.bsq"), dtype="<i2"
    )
    chikusei = raw.reshape(128, 2517, 2335)[:, 106:2410, 143:2191].astype(np.float32)
    yield "Chikusei", chikusei[:, :512, :512], [chikusei[:, 512:, :]]


def spatial_correlation(candidate, official):
    """Pearson correlation of band-mean images; both cubes must already be on the same 100-band grid."""
    a = candidate.mean(0).ravel()
    b = official.mean(0).ravel()
    if a.shape != b.shape:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_cube(path):
    """Read ``data`` from a MATLAB file (v5 or v7.3) as float32 (bands, rows, columns); bands is the 100-band axis."""
    try:
        from scipy.io import loadmat

        data = np.asarray(loadmat(path)["data"])
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as handle:
            data = np.asarray(handle["data"]).transpose()
    band_axis = int(np.argmin(data.shape))
    return np.moveaxis(data, band_axis, 0).astype(np.float32)


def official_test(root, name):
    files = sorted((root / "test" / name).glob("*.mat"))
    return load_cube(files[0]) if files else None


def build_training(root, seed=0):
    """Patch every training region whose hypothesised test crop matches the official one (correlation > 0.98)."""
    rng = np.random.default_rng(seed)
    chunks, sources, alignment = [], {}, {}
    for name, test_crop, regions in training_regions():
        official = official_test(root, name)
        if official is None:
            corr = float("nan")
        else:
            corr = spatial_correlation(minmax(to_100_bands(test_crop, name)), official)
        alignment[name] = corr
        print(
            f"[data] {name}: test-crop correlation with official file {corr:.4f}",
            flush=True,
        )
        if not corr > 0.98:
            print(
                f"[data] {name}: excluded from training (cannot prove the test crop is held out)",
                flush=True,
            )
            continue
        for i, region in enumerate(regions):
            cube = minmax(to_100_bands(region, name))
            block = patches(cube, rng)
            chunks.append(block)
            sources[f"{name}_{i}"] = {
                "shape": list(region.shape),
                "patches": len(block),
            }
            print(
                f"[data] {name}_{i}: region {region.shape} -> {len(block)} patches",
                flush=True,
            )
    data = np.concatenate(chunks)
    np.save(root / "train_patches.npy", data)
    summary = {
        "total": len(data),
        "regions": sources,
        "test_crop_correlation": alignment,
    }
    (root / "train_sources.json").write_text(json.dumps(summary, indent=1))
    print(
        f"[data] train_patches.npy {data.shape} {data.nbytes / 1e9:.2f} GB", flush=True
    )


def download_tests(root):
    import gdown

    files = gdown.download_folder(TEST_FOLDER, skip_download=True, quiet=True)
    for item in files:
        parts = item.path.replace("\\", "/").split("/")
        if len(parts) >= 3 and parts[-2] in TEST_SETS:
            target = root / "test" / parts[-2] / parts[-1]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                gdown.download(id=item.id, output=str(target), quiet=True)
            print(f"[data] test {parts[-2]}/{parts[-1]}", flush=True)
    ckpt = root / "Remote_sensing.ckpt"
    if not ckpt.exists():
        item = next(
            f
            for f in files
            if f.path.replace("\\", "/").endswith("ckpt/Remote_sensing.ckpt")
        )
        gdown.download(id=item.id, output=str(ckpt), quiet=True)
    print(f"[data] checkpoint {ckpt} {ckpt.stat().st_size / 1e6:.1f} MB", flush=True)


def build_real(root, size=512):
    """HyRank Hyperion scenes: first 50 of 176 bands = Hyperion VNIR bands 8-57 (426-925 nm) [assumption]."""
    import tifffile

    out = root / "real"
    out.mkdir(parents=True, exist_ok=True)
    for path in (
        "TrainingSet/Dioni.tif",
        "TrainingSet/Loukia.tif",
        "ValidationSet/Erato.tif",
    ):
        cube = tifffile.imread(hf_file("danaroth/hyrank", path)).astype(np.float32)
        cube = cube if cube.shape[0] == 176 else cube.transpose(2, 0, 1)
        _, rows, columns = cube.shape
        r0, c0 = max(0, (rows - size) // 2), max(0, (columns - size) // 2)
        crop = cube[:50, r0 : r0 + size, c0 : c0 + size]
        crop = crop[:, : crop.shape[1] // 64 * 64, : crop.shape[2] // 64 * 64]
        scene = minmax(to_100_bands(crop, "HyRank"), 0.1, 99.9)
        name = Path(path).stem
        np.save(out / f"{name}.npy", scene)
        print(f"[data] real {name} {scene.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    download_tests(args.root)
    if not args.skip_train and not (args.root / "train_patches.npy").exists():
        build_training(args.root)
    if not (args.root / "real").exists():
        build_real(args.root)


if __name__ == "__main__":
    main()
