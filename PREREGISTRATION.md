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

