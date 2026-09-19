"""Video <-> Open Ephys synchronisation (Phase 1).

Python port of ``synch_raw_*.m``. One self-contained module, laid out in the
same shape as the SSLH CLI: small private helpers at the top that everything
calls back to, then config dataclasses, then the readers as classes, then the
detection / fitting functions, then a single orchestrator, then the CLI.

Pipeline
--------
1. video LED ROI mean-intensity per frame -> detrend -> hysteresis rising-edge
   onset detection -> inter-pulse-interval cleanup
2. Open Ephys ADC ch99: LED modulates the *noise amplitude*, not a voltage
   level, so detection runs on a moving-std envelope with a locally adaptive
   median/MAD threshold, a sustained-"on" hold check, and a debounce
3. skip-aware monotonic pairing of the two pulse trains + linear fit
   ``ephys_time = a * video_time + b`` (iterated tolerance + outlier removal)

Outputs (into ``cfg.output_dir``)
    timebase.json         a, b, trim window, overlap, provenance
    timebase_frames.npz   tV, tEmap, tTrim  (per trimmed-video frame)
    sync_qc.json          pulse counts, residuals, pass/fail
    <video>_trimmed_to_ephys.mp4   (unless disabled)

Notes on fidelity to the MATLAB
    - channels are kept in raw ADC counts (no bit_volts, no downsampling),
      matching the 2026-07-31 FIX notes in the script
    - the 60 s local median/MAD threshold is computed on a decimated copy of
      the envelope and interpolated back; every threshold *crossing* and its
      sub-sample interpolation still happen at the native rate
    - moving mean/std use ``uniform_filter1d`` (edge mode "nearest"); MATLAB
      shrinks the window at the edges instead. Pulses within ``min_start_time_s``
      of the start are discarded anyway, so the edge difference does not reach
      the fit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a hard dep for real runs
    cv2 = None


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _win_split(k: int) -> tuple[int, int]:
    """(before, after) sample counts for a length-k centred window."""
    k = max(1, int(k))
    before = k // 2
    return before, k - 1 - before


def _movmean(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(x, dtype=np.float64)
    return uniform_filter1d(np.asarray(x, dtype=np.float64), size=int(k), mode="nearest")


def _movstd(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving standard deviation (population, matches MATLAB movstd w=1
    closely enough for thresholding; the fit does not depend on it)."""
    if k <= 1:
        return np.zeros_like(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()  # keep the squared term well inside float64
    m = uniform_filter1d(x, size=int(k), mode="nearest")
    m2 = uniform_filter1d(x * x, size=int(k), mode="nearest")
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _movmedian(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving median. Edge-padded; intended for small/modest arrays."""
    x = np.asarray(x, dtype=np.float64)
    if k <= 1 or x.size == 0:
        return x.copy()
    before, after = _win_split(k)
    xp = np.pad(x, (before, after), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, before + after + 1)
    return np.median(win, axis=1)


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation about the median (no 1.4826 scaling), == MATLAB mad(x,1)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(np.abs(x - np.median(x))))


def _block_reduce(x: np.ndarray, factor: int) -> np.ndarray:
    """Mean over consecutive blocks of ``factor`` samples (cheap anti-alias decimation)."""
    factor = max(1, int(factor))
    if factor == 1:
        return np.asarray(x, dtype=np.float64)
    n = (x.size // factor) * factor
    return x[:n].reshape(-1, factor).mean(axis=1)


def _crossing_time(y0: float, y1: float, t0: float, t1: float, thr: float) -> float:
    """Linear sub-sample time at which a segment (t0,y0)->(t1,y1) hits ``thr``."""
    if y1 == y0:
        return t1
    alpha = (thr - y0) / (y1 - y0)
    alpha = min(max(alpha, 0.0), 1.0)
    return t0 + alpha * (t1 - t0)


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv is required for video reading: pip install opencv-python-headless")


# ─────────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ChannelLayout:
    """Absolute 1-based indices within the continuous stream (MATLAB convention).

    Kept 1-based to match the scripts verbatim; converted at the numpy boundary.
    """

    eeg: list[int] = field(default_factory=lambda: list(range(1, 65)))
    emg: list[int] = field(default_factory=lambda: [65, 67])
    adc_led: int = 99
    n_channels_expected: int = 104


@dataclass
class SyncParams:
    """Header constants of ``synch_raw_*.m``."""

    # video onset
    ignore_first_seconds: float = 0.0
    smooth_median_frames: int = 5
    detrend_window_sec: float = 3.0
    min_peak_distance_s: float = 17.0
    thr_high_mad_mult_vid: float = 20.0
    thr_low_mad_mult_vid: float = 4.0
    min_on_hold_s: float = 0.05
    ipi_lo_frac: float = 0.6
    ipi_hi_frac: float = 1.4

    # ephys ADC envelope
    env_win_s: float = 0.10
    local_win_s: float = 60.0
    env_thr_mad_mult: float = 6.0
    min_isi_ephys: float = 17.0
    min_start_time_s: float = 1.0
    min_on_hold_s_ephys: float = 0.20
    max_above_frac: float = 0.10
    threshold_decim_hz: float = 20.0  # rate the adaptive threshold is computed at

    # pairing / fit
    initial_search_n: int = 6
    initial_search_tol_s: float = 3.0
    pairing_tolerances_s: list[float] = field(default_factory=lambda: [5.0, 3.0, 2.0])
    expected_ipi_s: float = 20.0
    outlier_floor_ms: float = 300.0
    outlier_mad_mult: float = 5.0

    # QC pass criteria
    min_matched_pulses: int = 8
    max_resid_median_ms: float = 50.0


@dataclass
class SessionConfig:
    session_id: str
    video_file: str
    oe_folder: str
    output_dir: str
    layout: ChannelLayout = field(default_factory=ChannelLayout)
    sync: SyncParams = field(default_factory=SyncParams)
    led_roi: list[int] | None = None  # [x, y, w, h] on the first frame
    write_trimmed_video: bool = True
    save_eeg_emg_adc: bool = True   # the EEG/EMG/ADC bundle Full_Analysis.m needs
    extract_eeg: bool = True        # matches her `extractEEG` -- off to skip the 64ch EEG and just keep EMG+ADC

    @classmethod
    def from_json(cls, path: str | Path) -> "SessionConfig":
        d = json.loads(Path(path).read_text())
        return cls(
            session_id=d["session_id"],
            video_file=d["video_file"],
            oe_folder=d["oe_folder"],
            output_dir=d["output_dir"],
            layout=ChannelLayout(**d.get("layout", {})),
            sync=SyncParams(**d.get("sync", {})),
            led_roi=d.get("led_roi"),
            write_trimmed_video=d.get("write_trimmed_video", True),
            save_eeg_emg_adc=d.get("save_eeg_emg_adc", True),
            extract_eeg=d.get("extract_eeg", True),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @property
    def out(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# ─────────────────────────────────────────────────────────────────────────────
# readers
# ─────────────────────────────────────────────────────────────────────────────
class RecordingReader:
    """Memory-mapped reader for Open Ephys 'binary' format (structure.oebin + continuous.dat)."""

    def __init__(self, oe_folder: str | Path, n_channels_expected: int = 104):
        self.oe_folder = Path(oe_folder)
        oebin = self.oe_folder / "structure.oebin"
        if not oebin.is_file():
            raise FileNotFoundError(f"structure.oebin not found in: {self.oe_folder}")
        meta = json.loads(oebin.read_text())

        streams = meta["continuous"]
        best = next(
            (s for s in streams if int(s.get("num_channels", -1)) == n_channels_expected),
            streams[0],
        )
        self.fs: float = float(best["sample_rate"])
        self.n_channels: int = int(best.get("num_channels", n_channels_expected))
        self.folder_name: str = best.get("folder_name", "")
        self.channel_names: list[str] = [c["channel_name"] for c in best.get("channels", [])]

        self.dat_path = self._find_dat()
        raw = np.memmap(self.dat_path, dtype="<i2", mode="r")
        if raw.size % self.n_channels != 0:
            if raw.size % n_channels_expected == 0:
                self.n_channels = n_channels_expected
            else:
                raise ValueError(
                    f"{self.dat_path.name} length {raw.size} not divisible by "
                    f"n_channels={self.n_channels}"
                )
        self.data = raw.reshape(-1, self.n_channels)  # samples x channels (lazy)
        self.n_samples = self.data.shape[0]

    def _find_dat(self) -> Path:
        hits = sorted(self.oe_folder.rglob("continuous.dat"))
        if not hits:
            raise FileNotFoundError(f"no continuous.dat under {self.oe_folder}")
        if self.folder_name:
            target = self.folder_name.rstrip("/")
            for h in hits:
                if target in str(h):
                    return h
        return hits[0]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def channel(self, idx_1based: int) -> np.ndarray:
        return np.asarray(self.data[:, idx_1based - 1], dtype=np.float64)

    def channels(self, idx_1based: list[int]) -> np.ndarray:
        """Several channels at once, kept as raw int16 counts (no float cast).
        For bulk per-channel exports (the EEG/EMG/ADC bundle) where dtype and
        size matter -- ``channel()`` casts to float64 because detection math
        needs it, but a 64-channel, full-recording export at float64 would be
        needlessly 4x the size of what the raw stream actually is."""
        cols = [i - 1 for i in idx_1based]
        return np.asarray(self.data[:, cols], dtype=np.int16)


@dataclass
class RoiTrace:
    t: np.ndarray            # per-frame timestamp (s)
    mean_i: np.ndarray       # raw ROI mean intensity
    mean_i_smooth: np.ndarray
    fps: float
    n_frames: int
    duration_s: float
    roi: tuple[int, int, int, int]


def read_roi_trace(video_file: str | Path, roi, smooth_median_frames: int = 5) -> RoiTrace:
    """Frame-by-frame ROI mean intensity with container timestamps (VIDEO block of the .m)."""
    _require_cv2()
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_file}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    x, y, w, h = (int(v) for v in roi)

    times: list[float] = []
    means: list[float] = []
    idx = 0
    while True:
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame = cap.read()
        if not ok:
            break
        t = t_ms / 1000.0 if t_ms and t_ms > 0 else idx / fps
        H, W = frame.shape[:2]
        patch = frame[max(0, y):min(y + h, H), max(0, x):min(x + w, W)]
        means.append(float(patch.mean()))
        times.append(t)
        idx += 1
    cap.release()

    mean_i = np.asarray(means, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    return RoiTrace(
        t=t,
        mean_i=mean_i,
        mean_i_smooth=_movmedian(mean_i, smooth_median_frames),
        fps=fps,
        n_frames=idx,
        duration_s=float(t[-1]) if t.size else 0.0,
        roi=(x, y, w, h),
    )


def read_first_frame(video_file: str | Path) -> np.ndarray:
    _require_cv2()
    cap = cv2.VideoCapture(str(video_file))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read first frame of {video_file}")
    return frame


def pick_roi_interactive(video_file: str | Path) -> tuple[int, int, int, int]:
    """Draw a tight box around the LED on the first frame (blocking UI)."""
    _require_cv2()
    frame = read_first_frame(video_file)
    r = cv2.selectROI("Draw a TIGHT box around the LED, then ENTER", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if r == (0, 0, 0, 0):
        raise RuntimeError("no ROI selected")
    return tuple(int(v) for v in r)


# ─────────────────────────────────────────────────────────────────────────────
# detection
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VideoDetection:
    times: np.ndarray
    n_raw: int
    n_removed_ipi: int
    median_ipi_s: float


def detect_video_led(roi: RoiTrace, p: SyncParams) -> VideoDetection:
    t, sig_raw = roi.t, roi.mean_i_smooth
    if p.ignore_first_seconds > 0:
        keep = t >= p.ignore_first_seconds
        t, sig_raw = t[keep], sig_raw[keep]

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("bad video timestamps")

    win = max(3, round(p.detrend_window_sec / dt))
    sig = sig_raw - _movmedian(sig_raw, win)

    sig_mad = _mad(sig)
    if sig_mad <= 0 or not np.isfinite(sig_mad):
        raise ValueError(f"invalid sig MAD ({sig_mad:.3g}); check ROI / detrend")
    thr_high = p.thr_high_mad_mult_vid * sig_mad
    thr_low = p.thr_low_mad_mult_vid * sig_mad

    hold = max(1, round(p.min_on_hold_s / dt))
    is_high = sig > thr_high
    rise_idx = np.flatnonzero(np.diff(np.concatenate(([0], is_high.astype(int)))) == 1)

    times: list[float] = []
    for j in rise_idx:
        j_end = min(sig.size - 1, j + hold)
        if np.count_nonzero(sig[j:j_end + 1] < thr_low) > 1:
            continue
        j0 = max(0, j - 1)
        t_cross = _crossing_time(sig[j0], sig[j], t[j0], t[j], thr_high)
        if not times or (t_cross - times[-1]) >= p.min_peak_distance_s:
            times.append(t_cross)

    times = np.asarray(times, dtype=np.float64)
    n_raw = times.size

    n_removed = 0
    if times.size >= 5:
        dv = np.diff(times)
        med = float(np.median(dv))
        lo, hi = p.ipi_lo_frac * med, p.ipi_hi_frac * med
        bad = np.zeros(times.size, dtype=bool)
        out = (dv < lo) | (dv > hi)
        bad[1:] |= out
        bad[:-1] |= out
        n_removed = int(np.count_nonzero(bad))
        times = times[~bad]

    med_ipi = float(np.median(np.diff(times))) if times.size >= 2 else float("nan")
    return VideoDetection(times, n_raw, n_removed, med_ipi)


@dataclass
class EphysDetection:
    times: np.ndarray
    n_raw_crossings: int
    n_rejected_hold: int
    n_removed_debounce: int
    median_ipi_s: float


def detect_ephys_pulses(adc: np.ndarray, fs: float, p: SyncParams) -> EphysDetection:
    n = adc.size
    tE = np.arange(n, dtype=np.float64) / fs

    env_win = max(3, round(p.env_win_s * fs))
    env = _movstd(adc, env_win)

    # locally adaptive median/MAD threshold, computed on a decimated envelope
    decim = max(1, int(round(fs / p.threshold_decim_hz)))
    env_ds = _block_reduce(env, decim)
    t_ds = (np.arange(env_ds.size) * decim + decim / 2.0) / fs
    local_win_ds = max(3, round(p.local_win_s * fs / decim))
    med_ds = _movmedian(env_ds, local_win_ds)
    mad_ds = _movmedian(np.abs(env_ds - med_ds), local_win_ds)
    thr_ds = med_ds - p.env_thr_mad_mult * mad_ds

    # clamp to the envelope's overall range, then lift back to native rate
    env_med, env_min = float(np.median(env)), float(env.min())
    thr_ds = np.clip(thr_ds, env_min, env_med)
    thr = np.interp(tE, t_ds, thr_ds, left=thr_ds[0], right=thr_ds[-1])

    is_on = env < thr
    fall = np.flatnonzero(np.diff(is_on.astype(np.int8)) == 1) + 1
    fall = fall[tE[fall] >= p.min_start_time_s]
    n_raw = fall.size

    # sustained-"on" hold check (vectorised via a cumulative count of env>=thr)
    hold = max(1, round(p.min_on_hold_s_ephys * fs))
    ge_cs = np.concatenate(([0], np.cumsum((env >= thr).astype(np.int64))))
    keep = []
    for j in fall:
        j_end = min(n, j + hold)
        win_len = j_end - j
        above = ge_cs[j_end] - ge_cs[j]
        if above <= max(1, round(p.max_above_frac * win_len)):
            keep.append(j)
    fall = np.asarray(keep, dtype=np.int64)
    n_rej_hold = n_raw - fall.size

    # sub-sample crossing time using the local threshold either side
    raw_times = np.empty(fall.size, dtype=np.float64)
    for i, j in enumerate(fall):
        thr_local = 0.5 * (thr[j - 1] + thr[j])
        raw_times[i] = _crossing_time(env[j - 1], env[j], tE[j - 1], tE[j], thr_local)

    # debounce against the most recently kept event
    times: list[float] = []
    n_deb = 0
    for tt in raw_times:
        if not times or (tt - times[-1]) > p.min_isi_ephys:
            times.append(tt)
        else:
            n_deb += 1
    times = np.asarray(times, dtype=np.float64)

    med_ipi = float(np.median(np.diff(times))) if times.size >= 2 else float("nan")
    return EphysDetection(times, n_raw, n_rej_hold, n_deb, med_ipi)


# ─────────────────────────────────────────────────────────────────────────────
# pairing + fit
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SyncFit:
    a: float
    b: float
    n_video_used: int
    n_ephys_used: int
    n_video_total: int
    n_ephys_total: int
    resid_ms_median: float
    resid_ms_max: float
    resid_ms_std: float
    tv_used: np.ndarray
    te_used: np.ndarray


def _initial_offset(tv: np.ndarray, te: np.ndarray, a_init: float, p: SyncParams) -> float:
    sn = min(p.initial_search_n, tv.size)
    sm = min(p.initial_search_n, te.size)
    best_cnt, best_b = -1, te[0] - a_init * tv[0]
    for ii in range(sn):
        for jj in range(sm):
            b_try = te[jj] - a_init * tv[ii]
            pred = a_init * tv + b_try
            cnt = int(np.count_nonzero(
                np.any(np.abs(te[None, :] - pred[:, None]) < p.initial_search_tol_s, axis=1)
            ))
            if cnt > best_cnt:
                best_cnt, best_b = cnt, b_try
    return best_b


def _walk_pairs(tv: np.ndarray, te: np.ndarray, a: float, b: float, tol: float):
    i = j = 0
    tv_u, te_u = [], []
    while i < tv.size and j < te.size:
        err = te[j] - (a * tv[i] + b)
        if abs(err) <= tol:
            tv_u.append(tv[i]); te_u.append(te[j]); i += 1; j += 1
        elif err < -tol:
            j += 1
        else:
            i += 1
    return np.asarray(tv_u), np.asarray(te_u)


def pair_and_fit(video_times: np.ndarray, ephys_times: np.ndarray, p: SyncParams) -> SyncFit:
    tv = np.asarray(video_times, dtype=np.float64)
    te = np.asarray(ephys_times, dtype=np.float64)
    if tv.size < p.min_matched_pulses or te.size < p.min_matched_pulses:
        raise ValueError(f"not enough pulses to sync (video={tv.size}, ephys={te.size})")

    a = float(np.median(np.diff(te)) / np.median(np.diff(tv)))
    b = _initial_offset(tv, te, a, p)

    tv_u = te_u = np.empty(0)
    resid_ms = np.empty(0)
    for k, tol in enumerate(p.pairing_tolerances_s):
        tv_u, te_u = _walk_pairs(tv, te, a, b, tol)
        if tv_u.size < p.min_matched_pulses:
            raise ValueError(f"too few matched pulses ({tv_u.size}) at tol={tol}")
        a, b = (float(v) for v in np.polyfit(tv_u, te_u, 1))
        resid_ms = (te_u - (a * tv_u + b)) * 1000.0

        if k > 0:
            thr_ms = max(p.outlier_floor_ms, p.outlier_mad_mult * float(np.median(np.abs(resid_ms))))
            outliers = np.abs(resid_ms) > thr_ms
            if outliers.any():
                tv_u, te_u = tv_u[~outliers], te_u[~outliers]
                a, b = (float(v) for v in np.polyfit(tv_u, te_u, 1))
                resid_ms = (te_u - (a * tv_u + b)) * 1000.0

    return SyncFit(
        a=a, b=b,
        n_video_used=tv_u.size, n_ephys_used=te_u.size,
        n_video_total=tv.size, n_ephys_total=te.size,
        resid_ms_median=float(np.median(np.abs(resid_ms))),
        resid_ms_max=float(np.max(np.abs(resid_ms))),
        resid_ms_std=float(np.std(resid_ms)),
        tv_used=tv_u, te_used=te_u,
    )


# ─────────────────────────────────────────────────────────────────────────────
# piecewise (anchor) mapping — fixes the dropped-frame sawtooth
# ─────────────────────────────────────────────────────────────────────────────
# A single global (a, b) assumes the video clock runs at one constant rate for
# the whole recording. If the camera drops frames it never writes to disk,
# v.CurrentTime (= frame_index / declared_fps) silently falls behind real time
# at each drop — invisible in the frame file itself, but visible as a sawtooth
# in the fit's residuals (steady drift, then a reset) rather than smooth drift.
# The fix: don't fit one line — walk the same robustly-matched pulse anchors
# from `pair_and_fit` and interpolate *between* them (piecewise-linear, linear
# extrapolation past the first/last anchor — matches MATLAB's
# interp1(...,'linear','extrap')). This self-corrects at every matched pulse
# regardless of how many frames were dropped between two of them.

@dataclass
class PiecewiseFit:
    anchors_video: np.ndarray
    anchors_ephys: np.ndarray

    @property
    def n_anchors(self) -> int:
        return self.anchors_video.size


def piecewise_fit(video_times: np.ndarray, ephys_times: np.ndarray, p: SyncParams) -> PiecewiseFit:
    """Same robust pulse matching as ``pair_and_fit``, kept as anchors instead of collapsed to one line."""
    fit = pair_and_fit(video_times, ephys_times, p)
    return PiecewiseFit(anchors_video=fit.tv_used, anchors_ephys=fit.te_used)


def map_video_to_ephys(t: np.ndarray, fit: "SyncFit | PiecewiseFit") -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if isinstance(fit, PiecewiseFit):
        ax, ay = fit.anchors_video, fit.anchors_ephys
        y = np.interp(t, ax, ay)
        if ax.size >= 2:
            slope_lo = (ay[1] - ay[0]) / (ax[1] - ax[0])
            left = t < ax[0]
            y[left] = ay[0] + slope_lo * (t[left] - ax[0])
            slope_hi = (ay[-1] - ay[-2]) / (ax[-1] - ax[-2])
            right = t > ax[-1]
            y[right] = ay[-1] + slope_hi * (t[right] - ax[-1])
        return y
    return fit.a * t + fit.b


def nn_residuals_ms(mapped_ephys_times: np.ndarray, ephys_pulse_times: np.ndarray) -> np.ndarray:
    """Nearest-neighbour residual (ms) of each mapped video pulse against the closest
    detected ephys pulse — evaluated across every detected pulse, not just the ones a
    fit chose to match, so drift/misalignment anywhere in the recording is visible."""
    mapped = np.asarray(mapped_ephys_times, dtype=np.float64)
    ref = np.asarray(ephys_pulse_times, dtype=np.float64)
    if ref.size == 0:
        return np.full(mapped.shape, np.nan)
    idx = np.clip(np.searchsorted(ref, mapped), 1, ref.size - 1)
    left, right = ref[idx - 1], ref[idx]
    nearest = np.where(np.abs(mapped - left) <= np.abs(mapped - right), left, right)
    return (nearest - mapped) * 1000.0


def piecewise_loo_residuals_ms(anchors_video: np.ndarray, anchors_ephys: np.ndarray) -> np.ndarray:
    """Honest quality check for the piecewise anchor mapping. Evaluating an anchor
    against the mapping it helped build is 0 by construction and proves nothing --
    instead, leave each anchor out one at a time, rebuild the mapping from the rest,
    and see how far off the held-out anchor's own video time lands. This is what
    QC's pass/fail is based on, not the trivial in-sample residual."""
    tv_u = np.asarray(anchors_video, dtype=np.float64)
    te_u = np.asarray(anchors_ephys, dtype=np.float64)
    resid = np.full(tv_u.size, np.nan)
    for i in range(tv_u.size):
        tv_rest, te_rest = np.delete(tv_u, i), np.delete(te_u, i)
        if tv_rest.size < 2:
            continue
        pw = PiecewiseFit(anchors_video=tv_rest, anchors_ephys=te_rest)
        resid[i] = map_video_to_ephys(np.array([tv_u[i]]), pw)[0] - te_u[i]
    return resid * 1000.0


def auto_find_led_roi(video_file: str | Path, sample_seconds: float = 90.0, box_size: int = 40):
    """Locate the LED without a human drawing a box: the LED is the pixel neighbourhood
    with the highest brightness variance over the first ``sample_seconds`` (everything
    else in frame is comparatively static). Streams frames (sum/sum-of-squares) rather
    than holding them all in memory."""
    _require_cv2()
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_file}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(round(sample_seconds * fps))

    s = s2 = None
    n = 0
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if s is None:
            s, s2 = g, g * g
        else:
            s += g
            s2 += g * g
        n += 1
    cap.release()
    if n < 10:
        raise RuntimeError(f"only read {n} frames; need at least 10 to auto-locate the LED")

    mean = s / n
    var = s2 / n - mean * mean
    var_smooth = uniform_filter1d(uniform_filter1d(var, 3, axis=0), 3, axis=1)
    y, x = np.unravel_index(np.argmax(var_smooth), var_smooth.shape)

    H, W = var.shape
    half = box_size // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    w, h = min(box_size, W - x0), min(box_size, H - y0)
    return (int(x0), int(y0), int(w), int(h))


# ─────────────────────────────────────────────────────────────────────────────
# orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class SyncRunner:
    """Runs the whole Phase-1 pipeline for one session and writes its outputs."""

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.p = cfg.sync

    def run(self, write_video: bool | None = None) -> dict:
        cfg, p = self.cfg, self.p
        roi = cfg.led_roi
        if roi is None:
            roi = pick_roi_interactive(cfg.video_file)

        roi_trace = read_roi_trace(cfg.video_file, roi, p.smooth_median_frames)
        vdet = detect_video_led(roi_trace, p)

        reader = RecordingReader(cfg.oe_folder, cfg.layout.n_channels_expected)
        adc = reader.channel(cfg.layout.adc_led)
        edet = detect_ephys_pulses(adc, reader.fs, p)

        fit = pair_and_fit(vdet.times, edet.times, p)
        a, b = fit.a, fit.b
        # Anchor mapping (piecewise-linear between matched pulses) is what actually
        # produces tEmap below -- it self-corrects at every anchor regardless of
        # dropped camera frames between them. The global (a, b) line is kept only
        # as a diagnostic: its own residual is what reveals a sawtooth when frames
        # were dropped, which the piecewise mapping is built to fix.
        pw = PiecewiseFit(anchors_video=fit.tv_used, anchors_ephys=fit.te_used)

        ephys_len = reader.duration_s
        t0 = max(0.0, (0.0 - b) / a)
        t1 = min(roi_trace.duration_s, (ephys_len - b) / a)
        if not t1 > t0:
            raise ValueError(f"invalid trim window [{t0:.3f}, {t1:.3f}]")

        frame_mask = (roi_trace.t >= t0) & (roi_trace.t <= t1)
        tV = roi_trace.t[frame_mask]
        tEmap = map_video_to_ephys(tV, pw)
        tTrim = np.arange(tV.size) / roi_trace.fps

        overlap_start = max(0.0, a * 0.0 + b)
        overlap_end = min(ephys_len, a * roi_trace.duration_s + b)

        global_resid_ms = nn_residuals_ms(map_video_to_ephys(vdet.times, fit), edet.times)
        pw_loo_resid_ms = piecewise_loo_residuals_ms(fit.tv_used, fit.te_used)
        pw_loo_median = float(np.nanmedian(np.abs(pw_loo_resid_ms))) if pw_loo_resid_ms.size else float("nan")
        pw_loo_max = float(np.nanmax(np.abs(pw_loo_resid_ms))) if pw_loo_resid_ms.size else float("nan")

        passed = (
            fit.n_video_used >= p.min_matched_pulses
            and pw_loo_median <= p.max_resid_median_ms
        )

        timebase = {
            "session_id": cfg.session_id,
            "mapping_method": "piecewise_anchored",
            "a": a,
            "b": b,
            "n_anchors": pw.n_anchors,
            "fs_ephys": reader.fs,
            "fps_video": roi_trace.fps,
            "trim_window_video_s": [t0, t1],
            "ephys_overlap_s": [overlap_start, overlap_end],
            "ephys_duration_s": ephys_len,
            "video_duration_s": roi_trace.duration_s,
            "n_trimmed_frames": int(tV.size),
            "led_roi_xywh": list(roi_trace.roi),
            "source_video": str(Path(cfg.video_file).resolve()),
            "source_oe_folder": str(Path(cfg.oe_folder).resolve()),
        }
        qc = {
            "session_id": cfg.session_id,
            "passed": bool(passed),
            "video": {
                "n_flashes": int(vdet.times.size),
                "n_raw": vdet.n_raw,
                "n_removed_ipi": vdet.n_removed_ipi,
                "median_ipi_s": vdet.median_ipi_s,
            },
            "ephys": {
                "n_pulses": int(edet.times.size),
                "n_raw_crossings": edet.n_raw_crossings,
                "n_rejected_hold": edet.n_rejected_hold,
                "n_removed_debounce": edet.n_removed_debounce,
                "median_ipi_s": edet.median_ipi_s,
            },
            # Diagnostic only -- one straight line for the whole recording. Its
            # residual is what exposes a dropped-frame sawtooth; it is not what
            # produced tEmap and is not what "passed" is judged on.
            "global_fit": {
                "a": a,
                "b": b,
                "n_matched": fit.n_video_used,
                "n_video_total": fit.n_video_total,
                "n_ephys_total": fit.n_ephys_total,
                "resid_ms_median": fit.resid_ms_median,
                "resid_ms_max": fit.resid_ms_max,
                "resid_ms_std": fit.resid_ms_std,
                "all_pulses_resid_ms_median": float(np.nanmedian(np.abs(global_resid_ms))) if global_resid_ms.size else float("nan"),
                "all_pulses_resid_ms_max": float(np.nanmax(np.abs(global_resid_ms))) if global_resid_ms.size else float("nan"),
            },
            # This is what actually produced tEmap, and what "passed" is judged on:
            # each anchor held out and predicted from its neighbours, not itself.
            "piecewise_fit": {
                "n_anchors": pw.n_anchors,
                "loo_resid_ms_median": pw_loo_median,
                "loo_resid_ms_max": pw_loo_max,
            },
            "expected_pulses": round(ephys_len / p.expected_ipi_s),
        }

        out = cfg.out
        (out / "timebase.json").write_text(json.dumps(timebase, indent=2))
        (out / "sync_qc.json").write_text(json.dumps(qc, indent=2))
        np.savez(out / "timebase_frames.npz", tV=tV, tEmap=tEmap, tTrim=tTrim)

        # EEG/EMG/ADC bundle for Full_Analysis.m's Python successor. synch_raw_*.m
        # already loaded these channels for LED-pulse detection (adc) -- this saves
        # the full set (+ EEG, + the two EMG channels) alongside the fit, same as
        # her own "EEG_EMG_ADC_downsampled_from_sync.mat" (raw int16 counts; the
        # name says "downsampled" but her own 2026-07-30 fix note says decimation
        # was removed -- this is full native-rate data, same as hers really is).
        if cfg.save_eeg_emg_adc:
            bundle_idx = (list(cfg.layout.eeg) if cfg.extract_eeg else []) + list(cfg.layout.emg) + [cfg.layout.adc_led]
            bundle = reader.channels(bundle_idx)
            n_eeg = len(cfg.layout.eeg) if cfg.extract_eeg else 0
            bundle_path = out / "EEG_EMG_ADC_from_sync.npz"
            np.savez(
                bundle_path,
                fs=reader.fs,
                n_samples=reader.n_samples,
                eeg=bundle[:, :n_eeg] if cfg.extract_eeg else np.empty((reader.n_samples, 0), dtype=np.int16),
                emg1=bundle[:, n_eeg],
                emg3=bundle[:, n_eeg + 1],
                adc3=bundle[:, n_eeg + 2],
                a=a, b=b,
                eeg_idx=np.asarray(cfg.layout.eeg, dtype=np.int32),
                emg_idx=np.asarray(cfg.layout.emg, dtype=np.int32),
                adc_idx=cfg.layout.adc_led,
            )
            timebase["eeg_emg_adc_bundle"] = str(bundle_path)
            (out / "timebase.json").write_text(json.dumps(timebase, indent=2))

        do_video = cfg.write_trimmed_video if write_video is None else write_video
        if do_video:
            trimmed = out / (Path(cfg.video_file).stem + "_trimmed_to_ephys.mp4")
            self._trim_video(cfg.video_file, trimmed, t0, t1)
            timebase["trimmed_video"] = str(trimmed)
            (out / "timebase.json").write_text(json.dumps(timebase, indent=2))

        return {"timebase": timebase, "qc": qc, "fit": fit, "piecewise_fit": pw}

    @staticmethod
    def _trim_video(src: str | Path, dst: str | Path, t0: float, t1: float) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; set write_trimmed_video=false to skip")
        # mp4/h264 -- not mjpeg-in-avi. A large (multi-GB) MJPEG/AVI file needs an
        # OpenDML extended index that not every OpenCV/ffmpeg build on Windows reads
        # the same way, which reads to a downstream tool (DeepLabCut) as "corrupted"
        # even when the file decodes perfectly cleanly (verified frame-by-frame both
        # via ffmpeg and cv2.VideoCapture). mp4/h264 is what DeepLabCut's own docs
        # recommend and is far more uniformly supported cross-platform.
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-ss", f"{t0:.6f}", "-to", f"{t1:.6f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
            "-an", str(dst),
        ]
        subprocess.run(cmd, check=True, capture_output=True)


def run_sync(cfg: SessionConfig, write_video: bool | None = None) -> dict:
    return SyncRunner(cfg).run(write_video=write_video)


# ─────────────────────────────────────────────────────────────────────────────
# cli
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Video <-> Open Ephys sync (Phase 1)")
    ap.add_argument("config", help="session config JSON")
    ap.add_argument("--no-video", action="store_true", help="skip writing the trimmed AVI")
    args = ap.parse_args(argv)

    cfg = SessionConfig.from_json(args.config)
    result = run_sync(cfg, write_video=False if args.no_video else None)
    qc = result["qc"]

    print(json.dumps(qc, indent=2))
    print(f"\n{'PASS' if qc['passed'] else 'FAIL'}  "
          f"anchors={qc['piecewise_fit']['n_anchors']}  "
          f"piecewise leave-one-out median={qc['piecewise_fit']['loo_resid_ms_median']:.2f} ms  "
          f"(old single-line fit median={qc['global_fit']['all_pulses_resid_ms_median']:.2f} ms)")
    print(f"outputs -> {cfg.out}")
    return 0 if qc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
