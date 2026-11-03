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

## Round 3, written 2026-11-02 after round 2 results and before any round 3 code was run

Round 2 left one gate effect standing in all four settings: fewer false alarms within 5 mm of
the tumour. Round 3 asks what produces it (H8) and whether gates help when a model meets a
different dataset (H9). Metrics, bootstrap and seed pooling are as in rounds 1 and 2.

### H8, what the gates do at the tumour edge (no new training)

Models: the six gated whole-tumour MSD models with Dice+BCE (InceptionV3 seeds 0-2, VGG-16
seeds 0-2), each paired with its plain partner where a comparison needs one. Test set: the 117
MSD test patients.

Gate profile. Each gate map is upsampled to 128 px. On every slice that contains tumour,
in-plane distance to the true tumour boundary is computed (1.75 mm pixels) and each brain pixel
is binned: inside tumour (>5 mm, 0-5 mm from the edge), outside (0-5 mm, 5-15 mm, >15 mm), plus
pixels outside the brain. Mean gate value per bin per patient.

- **H8a.** The finest gate suppresses the band just outside the tumour: per patient, mean gate
  value 0-5 mm outside the tumour minus mean gate value >15 mm from it is negative, with a CI
  excluding 0, in all six models. Other gate levels are reported, not scored.

Gate ablation. At test time every gate map is replaced by its own per-slice spatial mean, which
keeps the overall scale of the skip features and removes where the gate looks.

- **H8b.** Spatial gating causes the tighter edge: false-positive volume within 5 mm of the
  tumour rises under ablation (per-patient change, ablated minus intact, pooled over the three
  seeds, CI excludes 0) for both encoders. If ablation lowers an encoder's mean Dice by more
  than 0.05, H8b is scored uninterpretable for that encoder (the network is broken, not
  un-gated).
- Descriptive only: ablating one gate level at a time and setting all gates to 1, on seed 0 of
  each encoder; distant false alarms and Dice under every ablation.

### H9, gates across datasets

Models: InceptionV3 {plain, gates} x seeds {0, 1, 2}, Dice+BCE, 600 steps, trained on MSD exactly
as in round 1 except for the input channels, which are pre-contrast T1, FLAIR and post-contrast
T1 (T1gd) to match the LGG set. Test: all 110 LGG patients (none were seen in training), each
slice rotated 90 degrees to MSD orientation, with the LGG preprocessing of round 2 (z-score over
the whole head, skull and scalp kept). Predictions are rotated back and scored against the LGG
masks at the nominal LGG voxel size.

- **H9a.** Gates cut distant false alarms on the unseen dataset: seed-pooled mean per-patient
  change (gates minus plain) is negative with a CI excluding 0, and negative for every seed.
- **H9b.** Gates raise seed-pooled mean Dice on the unseen dataset by at least 0.01.
- Sensitivity analysis: both repeated without the 12 LGG patients whose pre- or post-contrast
  T1 was replaced by a copy of FLAIR in the source data.
- Descriptive only: the cost of the dataset shift (MSD-trained vs the round 2 LGG-trained models
  on the same 33 LGG test patients), the reverse direction (LGG-trained models on the 117 MSD
  test patients, rotated the other way), and the new MSD models' in-domain Dice against round 1.

## Round 4, written 2026-11-02 before any round 4 training

Every model so far trained for 600 steps of batch 16, about a quarter of an epoch, on a CPU.
Round 4 asks whether the main conclusions survive about 40 times that budget, on a Kaggle GPU.

Recipe: identical to round 1 (MSD whole tumour, FLAIR/T1gd/T2w, same slices and split, same
augmentation, AdamW 3e-4, cosine decay, batch 16, final weights, no checkpoint picking) except
24,500 steps (10 epochs), 300 warmup steps, and fp16 autocast with the loss in fp32. The slice
arrays are rebuilt on Kaggle from the same MSD tar and checked against SHA-256 hashes of the
local arrays before training. Run order, which is also the drop order if GPU time runs out
(runs are dropped from the end; hypotheses that lose a run are reported as not run):

1. VGG-16 {plain, gates} x seeds {0, 1, 2}, Dice+BCE
2. VGG-16 plain, Tversky, seeds {0, 1, 2}
3. InceptionV3 plain, Dice+BCE, seeds {0, 1, 2}
4. InceptionV3 with 2x input, plain, Dice+BCE, seeds {0, 1, 2}
5. InceptionV3 with 2x input, gates, Dice+BCE, seeds {0, 1, 2}

Each hypothesis is scored as in rounds 1 and 2 (per-patient paired differences, seed-pooled,
5,000-sample bootstrap), on the 117 MSD test patients.

- **H10a.** Gates raise seed-pooled mean Dice by at least 0.01. Scored separately for VGG-16 and
  for InceptionV3 at 2x input.
- **H10b.** Gates cut mean distant (>5 mm) false-positive volume: seed-pooled CI excludes 0 and
  every seed's mean change is negative. Scored per encoder as H10a.
- **H10c.** Gates cut mean false-positive volume within 5 mm of the tumour: seed-pooled CI
  excludes 0. (Exploratory in rounds 1 and 2, where it held in all four settings; now fixed in
  advance.) Scored per encoder.
- **H11.** A lower threshold buys what Tversky buys. Per seed, t* is the highest threshold on a
  1/255 grid at which VGG-16 plain Dice+BCE reaches, on the validation set, the mean sensitivity
  of VGG-16 plain Tversky at 0.5. On the test set the seed-pooled paired Dice difference
  (Dice+BCE at t* minus Tversky at 0.5) and mean sensitivity gap decide: "threshold is enough" if
  the CI lower bound is above -0.005 and the sensitivity gap is within 0.01; "Tversky adds
  something" if the CI upper bound is below 0; otherwise inconclusive. Metrics come from
  per-patient probability histograms (256 bins), as in H4.
- **H12a.** VGG-16 beats InceptionV3 at 1x: seed-averaged per-patient Dice difference (plain
  models) has a CI excluding 0.
- **H12b.** Input size explains the gap: InceptionV3 at 2x input closes at least half of the
  VGG-16 minus InceptionV3 1x gap in seed-mean Dice (plain models).
- Descriptive only: how much every configuration gains from the longer schedule, against its
  round 1 or 2 counterpart.
