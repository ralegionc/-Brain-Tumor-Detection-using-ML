import numpy as np
import pytest
import torch

from seg import losses
from seg.metrics import error_patterns, hausdorff, overlap, volume_metrics
from seg.models import SegNet


def cube(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


def test_overlap_known_values():
    a = cube((10, 10, 10), (0, 0, 0), (4, 4, 4))     # 64 voxels
    b = cube((10, 10, 10), (2, 0, 0), (6, 4, 4))     # 64 voxels, 32 shared
    m = overlap(a, b)
    assert m["dice"] == pytest.approx(0.5)
    assert m["iou"] == pytest.approx(32 / 96)
    assert m["sensitivity"] == pytest.approx(0.5)
    assert overlap(a, a)["dice"] == 1.0
    assert overlap(a, cube((10, 10, 10), (6, 6, 6), (9, 9, 9)))["dice"] == 0.0


def test_hausdorff_of_shift_equals_shift_in_mm():
    a = cube((20, 20, 20), (5, 5, 5), (10, 10, 10))
    b = np.roll(a, 3, axis=2)                          # shift 3 voxels along x
    assert hausdorff(a, b, (1, 1, 2.0), 100) == pytest.approx(6.0)
    assert hausdorff(a, a, (1, 1, 1), 95) == 0.0
    assert np.isnan(hausdorff(a, np.zeros_like(a), (1, 1, 1)))


def test_hd95_ignores_a_single_far_outlier():
    gt = cube((40, 40, 40), (10, 10, 10), (30, 30, 30))
    pred = gt.copy()
    pred[0, 0, 0] = True                               # one far false-positive voxel
    m = volume_metrics(pred, gt, (1, 1, 1))
    assert m["hd"] > 15 and m["hd95"] < 1.5


def test_error_patterns_separate_boundary_and_distant():
    gt = cube((30, 30, 30), (10, 10, 10), (20, 20, 20))
    pred = cube((30, 30, 30), (10, 10, 10), (21, 20, 20))   # one-voxel boundary overshoot
    pred |= cube((30, 30, 30), (0, 0, 0), (2, 2, 2))        # distant blob, 8 voxels
    e = error_patterns(pred, gt, (1, 1, 1))
    assert e["fp_boundary_ml"] == pytest.approx(100 / 1000)
    assert e["fp_distant_ml"] == pytest.approx(8 / 1000)
    assert e["fp_components"] == 1 and e["missed_lesions"] == 0
    assert e["fp_components_big"] == 0                      # 8 uL blob is under 0.1 mL
    assert error_patterns(pred, gt, (1, 1, 1), min_ml=0.005)["fp_components_big"] == 1
    gt2 = gt | cube((30, 30, 30), (25, 25, 25), (28, 28, 28))
    e2 = error_patterns(gt, gt2, (1, 1, 1))
    assert e2["missed_lesions"] == 1 and e2["missed_lesions_big"] == 0 and e2["gt_lesions_big"] == 1


@pytest.mark.parametrize("name", list(losses.LOSSES))
def test_losses_prefer_the_right_answer(name):
    t = torch.zeros(2, 1, 16, 16)
    t[:, :, 4:8, 4:8] = 1
    good = (t * 2 - 1) * 6
    bad = -good
    f = losses.LOSSES[name]
    assert f(good, t) < f(torch.zeros_like(t), t) < f(bad, t)


def test_tversky_penalises_misses_more_than_false_alarms():
    t = torch.zeros(1, 1, 16, 16)
    t[..., 4:12, 4:12] = 1                             # 64 tumour pixels
    miss = torch.full_like(t, -6.0)
    miss[..., 4:12, 4:8] = 6                           # finds only half: 32 misses
    alarm = (t * 2 - 1) * 6
    alarm[..., 0:4, 0:8] = 6                           # finds all, plus 32 false alarms
    assert losses.tversky(miss, t) > losses.tversky(alarm, t)
    # and widens the miss-vs-alarm gap relative to plain Dice
    tv = losses.tversky(miss, t) / losses.tversky(alarm, t)
    dc = losses.dice(miss, t) / losses.dice(alarm, t)
    assert tv > 1.5 * dc


@pytest.mark.parametrize("enc", ["vgg16", "vgg19", "inceptionv3"])
@pytest.mark.parametrize("att", [False, True])
def test_model_shapes_and_attention_maps(enc, att):
    m = SegNet(enc, att, pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 128, 128))
    assert y.shape == (2, 1, 128, 128)
    maps = m.attention_maps()
    assert len(maps) == (4 if att else 0)
    for a in maps:
        assert a.min() >= 0 and a.max() <= 1


def test_output_bias_starts_at_the_tumour_prior():
    # a forward-pass check is meaningless here: an untrained net in eval mode has default
    # batch-norm statistics, so its logits saturate whatever the bias
    m = SegNet("inceptionv3", True, pretrained=False)
    assert torch.sigmoid(m.head.bias).item() == pytest.approx(0.02)


def test_attention_net_overfits_a_tiny_batch():
    torch.manual_seed(0)
    m = SegNet("vgg16", True, pretrained=False, dec_ch=(64, 32, 16, 16))
    x = torch.randn(4, 3, 64, 64)
    t = (x[:, :1] > 0.8).float()                       # target is a function of the input
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(60):
        loss = losses.dice_bce(m(x), t)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        d = overlap(m(x).numpy() > 0, t.numpy() > 0)["dice"]
    assert d > 0.8
