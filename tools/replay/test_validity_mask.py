"""Gates for the validity mask (B4): unmasked must be IDENTICAL, masked must be real.

Run with the project environment, which is the one that has torch:

    ./.venv/bin/python -B -m unittest tools.replay.test_validity_mask

The contract at stake: a class a session could not measure arrives as an all-zero
target, so without a mask the loss trains "this class never exists" when the truth is
"this session never said". The change is only acceptable if a corpus that masks NOTHING
still trains exactly as before, so that half is checked as BIT-equality -- loss,
gradient and metrics -- and not as closeness.

The other half is checked on a deterministic stub model: a peak on an unsupervised
channel must not be counted at all, because counting it would charge the model for a
false positive against a truth nobody stated.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_minimap_heatmap as T  # noqa: E402

CLASSES = ("own", "lock")
GRID = 8
# A crop wide enough that the matching radius is well under one grid cell, so a peak
# either sits on the marker or misses it -- no accidental matches to reason about.
CROP = [0, 0, 720, 723]


class StubModel(torch.nn.Module):
    """Deterministic logits: one hot cell per channel, at (column 3, row 2) of the grid.

    ``hot_channels`` narrows which channels get that peak, which is how the
    "an unsupervised channel must not be counted" case is set up.
    """

    def __init__(self, classes=len(CLASSES), grid=GRID, rings=1, hot_channels=None,
                 at=(3, 2)):
        super().__init__()
        self.classes, self.grid, self.rings = classes, grid, rings
        self.hot_channels = tuple(hot_channels) if hot_channels is not None else None
        self.at = at
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images):
        size = images.shape[0]
        heat = torch.full((size, self.classes, self.grid, self.grid), -8.0)
        for channel in (self.hot_channels if self.hot_channels is not None
                        else range(self.classes)):
            heat[:, channel, self.at[1], self.at[0]] = 8.0
        ring = torch.full((size, self.rings), -8.0)
        return heat + self.bias * 0.0, ring


class Images:
    """The slice of ``LazyPictures`` evaluate() uses: torch float batches, not numpy."""

    def __init__(self, array):
        self.array = array

    def __getitem__(self, index):
        return torch.from_numpy(np.ascontiguousarray(self.array[index])).float()


class Tensors(Images):
    """The slice of ``LazyTargets`` evaluate() uses for rings and masks."""


def synthetic(samples=2, valid=None, ring_valid=None, negatives=None):
    """A minimal ``data`` dict for evaluate(): 2 classes, one marker on channel 0."""
    grid = GRID
    truth = np.full((samples, 2, 3), np.nan, dtype=np.float32)
    for row in range(samples):
        truth[row, 0] = (0.0, 3.0 / (grid - 1), 2.0 / (grid - 1))
    return {
        "classes": list(CLASSES),
        "grid": grid,
        "crop_rect": list(CROP),
        "truth": truth,
        "negatives": (np.zeros((samples, 2, 2), dtype=np.float32)
                      if negatives is None else negatives),
        "negative_counts": np.zeros(samples, dtype=np.int16) if negatives is None else
                           np.full(samples, len(negatives[0]), dtype=np.int16),
        "images": Images(np.zeros((samples, 3, 16, 16), dtype=np.float32)),
        "downsample": 1,
        "rings": Tensors(np.zeros((samples, 1), dtype=np.float32)),
        "ring_labels": [8.0],
        "valid": (np.ones((samples, len(CLASSES)), dtype=np.float32)
                  if valid is None else valid),
        "ring_valid": (np.ones(samples, dtype=np.float32)
                       if ring_valid is None else ring_valid),
    }


class UnmaskedIsUnchanged(unittest.TestCase):
    """With an all-ones mask the new code must reproduce the old one bit for bit."""

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4, 3, 8, 8)
        self.y = (torch.rand(4, 3, 8, 8) > 0.9).float()
        self.p = torch.tensor([2.5, 7.0, 1.0]).view(1, 3, 1, 1)
        self.ones = torch.ones(4, 3, 1, 1)

    def test_the_loss_is_bit_identical(self):
        reference = F.binary_cross_entropy_with_logits(self.x, self.y,
                                                       pos_weight=self.p, reduction="mean")
        got = T.masked_bce(self.x, self.y, self.ones, pos_weight=self.p)
        self.assertTrue(torch.equal(reference, got))

    def test_the_gradient_is_bit_identical(self):
        first = self.x.clone().requires_grad_(True)
        second = self.x.clone().requires_grad_(True)
        T.masked_bce(first, self.y, self.ones, pos_weight=self.p).backward()
        F.binary_cross_entropy_with_logits(second, self.y, pos_weight=self.p,
                                           reduction="mean").backward()
        self.assertTrue(torch.equal(first.grad, second.grad))

    def test_the_ring_loss_is_bit_identical(self):
        """The rings are masked per SAMPLE, so their mask has a different shape."""
        logits = torch.randn(4, 3)
        targets = (torch.rand(4, 3) > 0.5).float()
        reference = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        got = T.masked_bce(logits, targets, torch.ones(4, 1))
        self.assertTrue(torch.equal(reference, got))

    def test_the_metrics_are_identical_to_a_corpus_without_a_mask(self):
        index = np.asarray([0, 1])
        model = StubModel()
        with_mask, _, _ = T.evaluate(model, index, synthetic(), stream=True)
        without, _, _ = T.evaluate(model, index, synthetic(valid=None), stream=True)
        for name in CLASSES:
            for key, value in without[name].items():
                self.assertEqual(value, with_mask[name][key],
                                 "%s.%s changed when an all-ones mask was added"
                                 % (name, key))

    def test_the_identical_comparison_would_notice_a_change(self):
        """A harness that cannot fail proves nothing: the same comparison with the lock
        channel masked must differ from the all-ones run."""
        index = np.asarray([0, 1])
        model = StubModel()
        masked = np.ones((2, len(CLASSES)), dtype=np.float32)
        masked[:, 1] = 0.0
        with_mask, _, _ = T.evaluate(model, index, synthetic(valid=masked), stream=True)
        without, _, _ = T.evaluate(model, index, synthetic(), stream=True)
        self.assertNotEqual(without["lock"]["predicted"], with_mask["lock"]["predicted"])


class MaskedElementsAreExcluded(unittest.TestCase):
    """The half that actually changes behaviour."""

    def setUp(self):
        torch.manual_seed(1)
        self.x = torch.randn(4, 3, 8, 8)
        self.y = (torch.rand(4, 3, 8, 8) > 0.9).float()
        self.p = torch.tensor([2.5, 7.0, 1.0]).view(1, 3, 1, 1)

    def test_a_fully_masked_batch_is_zero_with_a_zero_gradient(self):
        logits = self.x.clone().requires_grad_(True)
        loss = T.masked_bce(logits, self.y, torch.zeros(4, 3, 1, 1), pos_weight=self.p)
        self.assertFalse(torch.isnan(loss))
        self.assertEqual(0.0, float(loss))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def assertSameValue(self, reference, got, note):
        """Equal to float noise, NOT bit for bit.

        A sum taken over a (N, C, H, W) tensor and the same sum taken over the kept
        channels alone accumulate in a different order, so the last bits differ. The
        bit-for-bit guarantee is reserved for the no-op case, where the canonical
        reduction is called and nothing is re-derived.
        """
        difference = float((torch.as_tensor(reference) - torch.as_tensor(got)).abs().max())
        self.assertLessEqual(difference, 1e-6,
                            "%s: max|difference| = %.3e" % (note, difference))

    def test_a_masked_channel_contributes_nothing(self):
        keep = torch.zeros(4, 3, 1, 1)
        keep[:, 0] = 1.0
        got = T.masked_bce(self.x, self.y, keep, pos_weight=self.p)
        reference = F.binary_cross_entropy_with_logits(
            self.x[:, 0:1], self.y[:, 0:1], pos_weight=self.p[:, 0:1], reduction="mean")
        self.assertSameValue(reference, got, "channel 0 kept, 1 and 2 masked")

    def test_masking_a_sample_removes_it_from_the_average(self):
        keep = torch.zeros(4, 3, 1, 1)
        keep[0] = 1.0
        keep[1] = 1.0
        got = T.masked_bce(self.x, self.y, keep, pos_weight=self.p)
        reference = F.binary_cross_entropy_with_logits(
            self.x[0:2], self.y[0:2], pos_weight=self.p, reduction="mean")
        self.assertSameValue(reference, got, "samples 2 and 3 masked")

    def test_an_unsupervised_channel_is_reported_and_counts_nothing(self):
        valid = np.ones((2, len(CLASSES)), dtype=np.float32)
        valid[:, 1] = 0.0
        metrics, _, _ = T.evaluate(StubModel(), np.asarray([0, 1]),
                                  synthetic(valid=valid), stream=True)
        self.assertTrue(metrics["own"]["supervised"])
        self.assertFalse(metrics["lock"]["supervised"])
        self.assertEqual(0, metrics["lock"]["predicted"])
        self.assertEqual(0, metrics["lock"]["truth"])
        self.assertIsNone(metrics["lock"]["precision"])

    def test_a_supervised_channel_still_counts(self):
        """The companion: the same stub DOES predict on `own`, so a supervised channel
        must keep its peak -- otherwise "counts nothing" would be trivially true."""
        metrics, _, _ = T.evaluate(StubModel(), np.asarray([0, 1]), synthetic(),
                                  stream=True)
        self.assertTrue(metrics["own"]["supervised"])
        self.assertEqual(2, metrics["own"]["predicted"])
        self.assertEqual(2, metrics["own"]["matched"])

    def test_a_channel_masked_in_one_session_only_is_still_supervised(self):
        """Availability is per session, and the split can hold both kinds: a channel
        supervised anywhere in the split must stay supervised there."""
        valid = np.ones((2, len(CLASSES)), dtype=np.float32)
        valid[1, 1] = 0.0
        metrics, _, _ = T.evaluate(StubModel(), np.asarray([0, 1]),
                                  synthetic(valid=valid), stream=True)
        self.assertTrue(metrics["lock"]["supervised"])
        self.assertEqual(1, metrics["lock"]["predicted"])


class PhantomRespectsTheMask(unittest.TestCase):
    """An unsupervised channel's peaks are neither right nor wrong.

    Measured on the real corpus the moment the mask landed: the phantom rate went
    0.318 -> 0.698, because zone_A/B/C and lock stopped being trained and their
    unconstrained peaks were still being counted as invented enemies.
    """

    # One unpainted-enemy position, exactly where the stub puts its peak.
    NEGATIVE = np.asarray([[[3.0 / (GRID - 1), 2.0 / (GRID - 1)]]] * 2, dtype=np.float32)

    def build(self, valid, hot):
        data = synthetic(valid=valid, negatives=self.NEGATIVE)
        # No true markers at all, or the position would be counted as EXPLAINED by one
        # and never judged -- the detail that made the first version of this gate pass
        # for the wrong reason.
        data["truth"] = np.full(data["truth"].shape, np.nan, dtype=np.float32)
        return data, StubModel(hot_channels=hot)

    def test_a_peak_from_an_unsupervised_channel_is_not_a_phantom(self):
        valid = np.ones((2, len(CLASSES)), dtype=np.float32)
        valid[:, 1] = 0.0
        data, model = self.build(valid, hot=(1,))          # only the masked channel peaks
        metrics, _, _ = T.evaluate(model, np.asarray([0, 1]), data, stream=True)
        phantom = metrics["phantom_peaks_on_unpainted_enemies"]
        self.assertEqual(2, phantom["judged"])             # one position, two samples
        self.assertEqual(0, phantom["phantom_hits"])
        self.assertEqual(0.0, phantom["share"])

    def test_the_same_peak_counts_once_that_channel_is_supervised(self):
        """The harness must be able to fail: with the channel supervised, that same peak
        IS a phantom. Without this, "counts nothing" would be trivially true."""
        data, model = self.build(None, hot=(1,))
        metrics, _, _ = T.evaluate(model, np.asarray([0, 1]), data, stream=True)
        phantom = metrics["phantom_peaks_on_unpainted_enemies"]
        self.assertEqual(2, phantom["judged"])             # one position, two samples
        self.assertEqual(2, phantom["phantom_hits"])       # one per sample


class RingsRespectTheMask(unittest.TestCase):
    """A session that never measured the rings must not be scored as "no ring enabled"."""

    def test_unsupervised_ring_samples_are_dropped(self):
        metrics, _, _ = T.evaluate(StubModel(), np.asarray([0, 1]),
                                  synthetic(ring_valid=np.zeros(2, dtype=np.float32)),
                                  stream=True)
        self.assertEqual({}, metrics["range_rings"])

    def test_supervised_ring_samples_are_kept(self):
        metrics, _, _ = T.evaluate(StubModel(), np.asarray([0, 1]), synthetic(),
                                  stream=True)
        self.assertIn("8.0", metrics["range_rings"])
        self.assertEqual(2, metrics["range_rings"]["8.0"]["samples"])

    def test_only_the_unsupervised_sample_is_dropped(self):
        metrics, _, _ = T.evaluate(
            StubModel(), np.asarray([0, 1]),
            synthetic(ring_valid=np.asarray([1.0, 0.0], dtype=np.float32)), stream=True)
        self.assertEqual(1, metrics["range_rings"]["8.0"]["samples"])


class PositiveWeightsRespectSupervision(unittest.TestCase):
    def weights(self, targets, valid, index=None, batch=2, cap=2000):
        data = {"classes": list(range(targets.shape[1])),
                "heatmaps": Tensors(targets), "valid": valid}
        if index is None:
            index = np.arange(len(targets))
        return T.positive_weights(data, index, batch, cap).flatten()

    def test_other_channels_do_not_multiply_the_negative_count(self):
        targets = np.zeros((2, 2, 2, 2), dtype=np.float32)
        targets[:, 0, 0, 0] = 1
        targets[:, 1, 0, :] = 1
        actual = self.weights(targets, np.ones((2, 2), dtype=np.float32))
        self.assertTrue(torch.equal(actual, torch.tensor([3., 1.])))
        single = self.weights(targets[:, :1], np.ones((2, 1), dtype=np.float32))
        self.assertEqual(actual[0], single[0])

    def test_unsupervised_targets_and_nontraining_samples_do_not_count(self):
        targets = np.zeros((3, 2, 2, 2), dtype=np.float32)
        targets[0, :, 0, 0] = 1
        targets[1, 0] = 1  # Unsupervised: neither pixels nor positives may count.
        targets[2] = 1     # Held out: must not affect training weights.
        valid = np.asarray([[1, 1], [0, 1], [1, 1]], dtype=np.float32)
        actual = self.weights(targets, valid, np.asarray([0, 1]), batch=1)
        self.assertTrue(torch.equal(actual, torch.tensor([3., 7.])))

    def test_streaming_and_single_sample_tail_match_whole_batch(self):
        targets = np.zeros((3, 2, 2, 2), dtype=np.float32)
        targets[:, 0, 0, 0] = 1
        targets[:, 1, 0, :] = 1
        valid = np.ones((3, 2), dtype=np.float32)
        self.assertTrue(torch.equal(self.weights(targets, valid, batch=2),
                                    self.weights(targets, valid, batch=3)))

    def test_empty_positive_channel_is_capped_and_unsupervised_is_zero(self):
        targets = np.zeros((2, 2, 2, 2), dtype=np.float32)
        valid = np.asarray([[1, 0], [1, 0]], dtype=np.float32)
        self.assertTrue(torch.equal(self.weights(targets, valid, cap=5),
                                    torch.tensor([5., 0.])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
