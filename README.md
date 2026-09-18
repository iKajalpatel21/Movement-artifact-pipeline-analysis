# pup_state_pipeline

Python port of Evelina's MATLAB pup-recording pipeline: sync ThorCam video to
Open Ephys, derive LFP/EMG/movement signals, classify behavioural/arousal state
(Control / Jerk / Startle / Aroused), and auto-generate a per-session report.

Standalone desktop tool (Qt UI planned) — **not** part of SpikeSortingLabHub.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Sync: video LED ↔ ADC pulses → `ephys_time = a·video_time + b` | in progress |
| 2 | Downsample 30 kHz → 1 kHz derived matrix | not started |
| 3 | Full analysis (LFP/spikes/eSPW/DLC/EMG) + arousal classification | not started |
| 4 | Auto report per animal/session | not started |

Reference MATLAB lives on the NAS at `/Volumes/experiments/EB_backup/AD086/`.

## Layout

```
pup_state_pipeline/
  config.py          SessionConfig — per-session paths + channel layout + tuning constants
  io/
    openephys.py     RecordingReader — memory-mapped continuous.dat + structure.oebin
    video.py         frame-by-frame ROI brightness with real timestamps
  sync/
    detect.py        video LED onset + ADC envelope pulse detection
    fit.py           robust monotonic pairing + linear fit
    run.py           orchestrator: produces timebase + QC json
configs/             one JSON per session
scripts/             thin CLI entry points
tests/               correctness gates (Python a,b vs MATLAB a,b)
```

## Dev

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_sync.py configs/AD086_exp1rec1.json
```
