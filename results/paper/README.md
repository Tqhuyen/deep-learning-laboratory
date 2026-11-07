# The denoising paper package

Three files, in the order a reviewer should read them.

| file | what it is |
|---|---|
| [`claims.md`](claims.md) | Every claim the paper makes, with its evidence, its statistic and the result that would falsify it. Also lists what is explicitly **not** claimed, and the three conditions under which the headline claims would be withdrawn. |
| [`draft.md`](draft.md) | The paper: abstract, introduction, related work, method, protocol, results, limitations. Mechanism claims that were not tested are marked "not run" or "not separated". |
| [`prior_art.md`](prior_art.md) | Every component of the method has prior art, with what that work established and what it did not do. Also the two gaps this paper fills, and the one comparison that is impossible because no weights were released. |

## The evidence

Each claim resolves to a `results/<study>/README.md` and the CSVs beside it. The studies:

- [`../diffusion_benchmark/`](../diffusion_benchmark/README.md): one-step posterior mean on the three small colour sets,
  what the remaining knobs add, and the pre-registered test of DMID's own multi-evaluation chain.
- [`../gray_lift/`](../gray_lift/README.md): grayscale through an exact lift, with released supervised models rerun on
  identical draws.
- [`../urban100/`](../urban100/README.md) and [`../urban100/full100/`](../urban100/full100/README.md): large images,
  tiles, and the paired comparison against GRL-B's released outputs; [`../urban100/seed_replication/`](../urban100/seed_replication/README.md)
  for the three-draw replication.
- [`../adaptation/`](../adaptation/README.md): per-image test-time adaptation, its failures, and the one measured
  quantity that predicts when it helps.
- [`../measurement_audit/`](../measurement_audit/README.md): how much the scoring protocol costs, measured on two
  architectures with the denoiser otherwise held fixed.
- [`../real_noise/`](../real_noise/README.md): camera noise, reported as analysis rather than as a claim.

## Reproducing

Every result names the Kaggle kernel that produced it and the notebook source in [`../../notebooks/`](../../notebooks).
`kaggle_output/` holds the notebook sources and build directories and is gitignored; the kernels themselves are public
under the account referenced in each report.

The decision log that governed which runs were allowed — pre-registrations, amendments, and the runs that were killed —
is not in this repository; its outcomes are recorded in each study's README and in [`claims.md`](claims.md).
