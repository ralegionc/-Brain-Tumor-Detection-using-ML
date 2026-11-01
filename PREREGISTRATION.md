# Pre-registration

Written 2026-10-30, before any model was trained. Not edited after results came in;
any later change goes in a dated addendum at the bottom.

## Setup

- Data: Medical Segmentation Decathlon Task01 (484 BraTS 2016/17 volumes, 4 MRI sequences).
  Inputs are FLAIR, T1gd and T2w; the target is whole tumour (oedema + non-enhancing + enhancing).
- Patient-level split by hash: 314 train, 53 val, 117 test. Test is touched once per run.
- 2D axial slices at 128 x 128 (1.75 mm pixels), U-Net decoders on ImageNet encoders
  (VGG-16-BN, VGG-19-BN, InceptionV3), with or without additive attention gates on the skips.
- Every run: 600 steps, batch 16, AdamW 3e-4, cosine decay, final weights (no checkpoint picking).
- Metrics are per 3D test volume: Dice, IoU, sensitivity, precision, HD95, HD, plus error patterns:
  false-positive volume within 5 mm of the true tumour (boundary) and beyond it (distant),
  false-positive components touching no tumour, and missed lesions.

## Hypotheses

**H1a, attention and false alarms.** Attention gates reduce distant false-positive volume
per patient compared with the same encoder without gates. Supported if the paired median
difference is negative for all three encoders, and the bootstrap 95% CI excludes 0 for at
least two of them and for the pooled InceptionV3 seed replicates.

**H1b, attention and overlap.** Attention gates raise mean Dice by at least 0.01 over the
matching plain model. Supported if this holds for at least two of three encoders and for
the InceptionV3 seed average.

**H2a, Tversky trades precision for sensitivity.** Tversky (alpha 0.3, beta 0.7) and focal
Tversky give higher mean sensitivity and lower mean precision than Dice+BCE.

**H2b, plain BCE under-segments.** Plain BCE gives the lowest mean sensitivity of the six losses.

**H3, encoders.** No directional prediction. Reported descriptively.

## Noise floor

Each configuration outside phase C has one training seed, so a test-set CI covers patient
sampling but not training randomness. The spread of the three InceptionV3 seeds (with and
without gates) is the yardstick: a difference between single-seed runs smaller than that
spread is reported as not distinguishable from seed noise, whatever its CI says.

## Addendum 2026-10-30, before any test result

The first queued run was stopped at step 225 of 600. Its validation Dice looked fine (0.77),
but background pixels sat at probability ~0.25 because the output bias started at 0 and
Adam at 3e-4 moves it only ~0.07 in 225 steps. The output bias now starts at the tumour
prior (log(0.02 / 0.98)), as in the focal-loss paper. A 90-step check reached validation
Dice 0.853 with background probability 0.015. All runs restart from scratch with this
initialisation; nothing else in the setup or hypotheses changed, and no test volume had
been evaluated.

## Round 2, written 2026-10-31 after round 1 results and before any round 2 training or evaluation

Round 1 left four questions open. Each is tested below with the same training budget
(600 steps, batch 16) and the same per-patient paired bootstrap unless stated.

### H4, is a loss function just a threshold?

Tversky at threshold 0.5 found more tumour than Dice+BCE (sensitivity 0.910 vs 0.861) at a
cost in precision. A plainer route to the same trade-off is to lower Dice+BCE's threshold.
Both gated InceptionV3 seed-0 models are re-run on the validation and test sets with full
probability maps. On the **validation** set, pick t*, the threshold at which Dice+BCE's mean
sensitivity first reaches Tversky's mean validation sensitivity at 0.5. Then on the **test** set,
compare Dice+BCE at t* with Tversky at 0.5:

- **"Threshold is enough"** if the paired mean Dice difference (Dice+BCE at t* minus Tversky)
  has a 95% CI lower bound above -0.005 and the two test sensitivities are within 0.01.
- **"Tversky adds something"** if the CI upper bound is below 0.
- Otherwise inconclusive. The full sensitivity and precision curves are descriptive.

### H5, sub-regions: enhancing tumour

Enhancing tumour (label 3) is smaller and more imbalanced than whole tumour. Same slices,
target swapped. Runs: InceptionV3 {plain, gates} x {Dice+BCE, Tversky}, seed 0, plus
{plain, gates} Dice+BCE seed 1 as a noise floor. Patients with no enhancing tumour score
Dice 1 if the prediction is empty and 0 otherwise (BraTS convention), and are counted.

- **H5a.** Gates raise enhancing-tumour Dice by at least 0.01 over plain, for both losses and
  for the seed-pooled Dice+BCE pair.
- **H5b.** Tversky raises enhancing-tumour Dice over Dice+BCE (paired CI excludes 0) for both
  the plain and the gated model. If it only trades precision for sensitivity again, H5b fails.

### H6, attention gates on scans that keep the skull

BraTS is skull-stripped, so the "fat tissue" false positives attention gates are meant to
suppress cannot occur there. The LGG segmentation set (Buda et al. 2019, 110 TCGA lower-grade
glioma patients, pre-contrast T1, FLAIR, post-contrast T1, FLAIR-abnormality masks) is not
skull-stripped: scalp fat and skull are in every slice. Split by patient hash: 30% test,
10% validation, 60% train. Slices are 256 px resampled to 128 px; the source pixel spacing is
not stored in the files, so volumes use a nominal 1.9 x 1.9 x 3 mm voxel and are only compared
within this dataset. Runs: InceptionV3 {plain, gates} x seeds {0, 1, 2}, Dice+BCE.

Because round 1 showed the median distant false-positive change is pinned at 0 when many
patients have none, H6 uses the mean.

- **H6a.** Gates cut distant (>5 mm from tumour) false-positive volume: the seed-pooled mean
  per-patient change is negative with a CI excluding 0, and the mean is negative for each seed.
- **H6b.** Gates raise seed-pooled mean Dice by at least 0.01.

### H7, encoders, replicated

- **H7a.** VGG-16 beats InceptionV3 on whole-tumour Dice with three seeds each (VGG-16 seeds 1
  and 2, plain and gated, are added): the seed-averaged per-patient difference has a CI that
  excludes 0 for both plain and gated models.
- **H7b.** InceptionV3's deficit is about input resolution. InceptionV3 plain, Dice+BCE, seed 0
  is retrained with each 128 px slice upsampled 2x inside the network (prediction downsampled
  back, so labels and metrics are unchanged). Supported if its test Dice closes at least half
  the round 1 gap to VGG-16 plain seed 0, that is reaches 0.877 (InceptionV3 0.868, VGG-16 0.885).
- The VGG-16 seed runs also give a three-seed check of H1 on VGG-16 (descriptive).

