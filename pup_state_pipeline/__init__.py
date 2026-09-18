"""Python port of Evelina's pup-recording pipeline."""

from .Synchronization import (
    ChannelLayout,
    SessionConfig,
    SyncParams,
    SyncRunner,
    RecordingReader,
    detect_video_led,
    detect_ephys_pulses,
    pair_and_fit,
    piecewise_fit,
    PiecewiseFit,
    map_video_to_ephys,
    nn_residuals_ms,
    piecewise_loo_residuals_ms,
    auto_find_led_roi,
    run_sync,
)

__version__ = "0.0.1"

__all__ = [
    "ChannelLayout",
    "SessionConfig",
    "SyncParams",
    "SyncRunner",
    "RecordingReader",
    "detect_video_led",
    "detect_ephys_pulses",
    "pair_and_fit",
    "piecewise_fit",
    "PiecewiseFit",
    "map_video_to_ephys",
    "nn_residuals_ms",
    "piecewise_loo_residuals_ms",
    "auto_find_led_roi",
    "run_sync",
]
