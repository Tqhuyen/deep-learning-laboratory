# Tile rule R on the small colour sets (CPU-free GPU run)

Run: Kaggle GPU kernel [`trnquanghuyn/tile-rule-colour`](https://www.kaggle.com/code/trnquanghuyn/tile-rule-colour)
(T4 ×2) of [`notebooks/tile_rule_colour.ipynb`](../../../notebooks/tile_rule_colour.ipynb). Nothing was selected on these
images (amendment 6): rule R is 384-px tiles with overlap 96, chosen on Urban100 dev-12, applied unchanged.

- **Scores:** [`scores.csv`](scores.csv) per image, σ, arm and flip.
- **Tests:** [`paired.csv`](paired.csv), per-set paired contrasts with a sign-flip test and Holm correction.

## Result: tiles-384 ×2 − whole ×2 (dB)

| set | σ 15 | σ 25 | σ 50 |
|---|---:|---:|---:|
| Kodak24 (n = 24) | +0.017 | +0.019 | +0.017 |
| McMaster (n = 18) | +0.010 | +0.011 | +0.002 |
| CBSD68 (n = 68) | — | −0.005 | −0.019 |

Rule R is worth at most +0.02 dB on these sets and is slightly negative on CBSD68 at σ 50. The tile rule is therefore a
large-image tool only, and it is not a contribution on the small sets.

## P1c: DMID's own three-evaluation chain at η = 0 against one step

The same run evaluated DMID's S_t = 2 chain (seq `[0, ⌊N/2⌋, N−1]`, η = 0, three evaluations) on the whole image, with
DMID's integer N, as a pre-registered test of the "training-free knobs are exhausted" claim.

| σ | chain ×2 − one step ×2 | images better | n |
|---|---:|---:|---:|
| 15 | **−0.128** | 0/110 | 110 |
| 25 | **−0.137** | 0/110 | 110 |
| 50 | **−0.114** | 0/110 | 110 |

**Verdict: not falsified.** DMID's own multi-evaluation chain is 0.11–0.14 dB *worse* than one step on Gaussian noise,
on every image at every σ.

This closes two open items:

- The chain that carried DMID-d on real noise ([`../../real_noise`](../../real_noise/README.md)) buys nothing on
  matched Gaussian noise.
- The earlier "two steps" ablation (t → 0 → clean) was not DMID's chain, as the Phase-1 audit found. With the correct
  chain the conclusion is stronger, not weaker.
