"""Phase-1 sync tests.

Unit tests run anywhere. The AD086 integration tests are skipped unless the NAS
is mounted at /Volumes/experiments and the MATLAB reference outputs are present.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pup_state_pipeline.Synchronization import (
    SyncParams,
    RecordingReader,
    detect_ephys_pulses,
    pair_and_fit,
    _mad,
    _movmedian,
    _movstd,
)

AD086_OE = Path(
    "/Volumes/experiments/EB_backup/AD086/2026-08-17_15-39-04/Record Node 102/experiment1/recording1"
)
AD086_REF_MAT = Path(
    "/Volumes/experiments/EB_backup/AD086/AD086_exp1rec1/video_to_ephys_timebase.mat"
)
# NAS reads pull the whole ~15 GB continuous.dat over the network (one interleaved
# channel = every page), so these are opt-in: PUP_RUN_NAS=1 pytest
nas = pytest.mark.skipif(
    not (AD086_OE.exists() and os.environ.get("PUP_RUN_NAS") == "1"),
    reason="set PUP_RUN_NAS=1 with the AD086 NAS mounted to run",
)


# ── helpers ──────────────────────────────────────────────────────────────────
def test_mad_matches_definition():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert _mad(x) == pytest.approx(1.0)  # median |x - 3| = median([2,1,0,1,97])


def test_movmedian_odd_window_is_centred():
    x = np.array([0.0, 0.0, 9.0, 0.0, 0.0])
    assert _movmedian(x, 3).tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_movstd_constant_signal_is_zero():
    assert np.allclose(_movstd(np.full(1000, 7.0), 51), 0.0)


# ── pairing + fit ────────────────────────────────────────────────────────────
def _synthetic_trains(a=1.0002, b=12.34, n=40, ipi=20.0, seed=0):
    rng = np.random.default_rng(seed)
    tv = np.cumsum(rng.normal(ipi, 0.05, n)) + 5.0
    te = a * tv + b + rng.normal(0.0, 0.003, n)  # ~3 ms jitter
    return tv, te


def test_fit_recovers_known_mapping():
    tv, te = _synthetic_trains()
    fit = pair_and_fit(tv, te, SyncParams())
    assert fit.a == pytest.approx(1.0002, abs=1e-4)
    assert fit.b == pytest.approx(12.34, abs=0.1)
    assert fit.resid_ms_median < 5.0
    assert fit.n_video_used >= 38


def test_fit_survives_dropped_and_extra_pulses():
    tv, te = _synthetic_trains(seed=1)
    tv = np.delete(tv, [7, 21])                     # video missed two
    te = np.insert(te, 15, te[15] - 8.0)            # ephys has a spurious extra
    fit = pair_and_fit(tv, te, SyncParams())
    assert fit.a == pytest.approx(1.0002, abs=2e-4)
    assert fit.resid_ms_median < 10.0


def test_fit_rejects_too_few_pulses():
    with pytest.raises(ValueError):
        pair_and_fit(np.arange(5) * 20.0, np.arange(5) * 20.0, SyncParams())


# ── AD086 integration ────────────────────────────────────────────────────────
@nas
def test_recording_reader_opens_ad086():
    r = RecordingReader(AD086_OE)
    assert r.fs == 30000.0
    assert r.n_channels == 104
    assert r.duration_s > 60


@nas
def test_ephys_pulse_count_near_expected():
    r = RecordingReader(AD086_OE)
    det = detect_ephys_pulses(r.channel(99), r.fs, SyncParams())
    expected = round(r.duration_s / 20.0)
    assert abs(det.times.size - expected) <= max(3, 0.1 * expected)
    assert 18.0 < det.median_ipi_s < 22.0


@nas
@pytest.mark.skipif(not AD086_REF_MAT.exists(), reason="reference .mat missing")
def test_reference_mat_is_readable():
    from scipy.io import loadmat

    m = loadmat(AD086_REF_MAT)
    assert "a" in m and "b" in m
    # once led_roi is filled into configs/AD086_exp1rec1.json, assert our fit
    # matches m["a"], m["b"] here.
