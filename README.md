# pup_state_pipeline

A Python port of the MATLAB pipeline used to process neonatal-mouse pup
recordings (video + electrophysiology) for arousal-state research —
synchronizing raw video and ephys, then (in later phases) deriving
LFP/EMG/movement signals and classifying behavioral/arousal state
(Control / Jerk / Startle / Aroused).

Standalone tool, built to eventually run as a desktop app — **not** part of
SpikeSortingLabHub, and not a replacement for the underlying science, just a
faster, more maintainable way to run it.

## Why this exists

The original pipeline is a set of MATLAB scripts (`Synch_raw.m`,
`Full_Analysis.m`, …) run by hand, one recording at a time. Each script only
works on the machine it was written on, has no automated tests, and produces
no machine-checkable pass/fail — sync quality is judged by eyeballing a plot.

This project ports the pipeline stage-by-stage into Python modules named
after the MATLAB scripts they replace, so it stays recognizable to anyone
who knows the original, while gaining:
- **real validation** — every stage is checked against real recordings and
  the original MATLAB's own output, not synthetic data
- **honest QC** — a machine-readable pass/fail per session, not a plot to
  eyeball
- **portability** — runs anywhere Python runs, no MATLAB license required

## Pipeline overview

<!-- Diagram goes here. -->
<!-- ![Pipeline overview](docs/pipeline_diagram.png) -->

The pipeline follows the original MATLAB structure:

```
Raw experiment
 ├─ Open Ephys (structure.oebin + continuous.dat)
 └─ Video (raw .avi)
        │
        ▼
 Synchronization  →  timebase (video↔ephys time mapping) + trimmed video
        │                     + EEG/EMG/ADC channel export
        ▼
 DeepLabCut (pose estimation — external, unchanged)  →  tracked CSV + labeled video
        │
        ▼
 Full analysis (LFP / spikes / eSPW / EMG / arousal classification)
        │
        ▼
 Per-session figures, correlations, PETHs, event tables
```

## Status

| Phase | What | State |
|---|---|---|
| 1 | **Synchronization** — video LED ↔ ephys ADC pulses → time mapping, trimmed video, EEG/EMG/ADC export | ✅ done, validated on real data |
| 2 | Pose estimation (DeepLabCut) integration | external tool, wiring not started |
| 3 | Full analysis — LFP/spikes/eSPW/EMG + arousal-state classification | not started |
| 4 | Auto-generated per-session report | not started |

### Phase 1 validation

Synchronization was validated against a real recording, not synthetic data.
The original approach fits a single straight line for the whole recording
(`ephys_time = a·video_time + b`), which breaks down whenever the camera
drops a frame — the drop is invisible in the video file itself, but shows up
as drift between video and ephys later in the recording.

| | Single global fit | Piecewise-anchored fit |
|---|---|---|
| Median timing error | 102.4 ms | **12.2 ms** |
| Max timing error | 266.4 ms | 122.5 ms |

The piecewise result above is a **leave-one-out** estimate — each matched
pulse is held out, the mapping is rebuilt from the rest, and the held-out
pulse's predicted time is checked against its real one. Evaluating a fit
against the same points used to build it is circular and was deliberately
avoided.

## Layout

```
pup_state_pipeline/
    __init__.py
    Synchronization.py     Phase 1: video<->ephys sync, trimmed video, EEG/EMG/ADC export
configs/                   one JSON session config per recording
tests/                     unit tests + NAS-gated integration tests
out/                       example output (comparison chart)
```

One module per pipeline stage, named after the MATLAB script it replaces —
deliberately kept to a small number of files rather than split into many
small ones.

## Session config

Each recording gets one JSON config describing its inputs and channel layout:

```json
{
  "session_id": "AD087_exp1rec1",
  "video_file": "/path/to/raw_video.avi",
  "oe_folder": "/path/to/open_ephys/recording1",
  "output_dir": "out/AD087_exp1rec1",
  "led_roi": [623, 171, 50, 32],
  "write_trimmed_video": true,
  "save_eeg_emg_adc": true,
  "extract_eeg": true,
  "layout": {
    "eeg": [1, 2, "...", 64],
    "emg": [65, 67],
    "adc_led": 99,
    "n_channels_expected": 104
  }
}
```

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m pup_state_pipeline.Synchronization configs/AD087_exp1rec1.json
```

### Outputs (written to `output_dir`)

| File | What |
|---|---|
| `timebase.json` | fit coefficients, trim window, provenance — human-readable |
| `sync_qc.json` | pulse counts, residuals, pass/fail |
| `timebase_frames.npz` | per-frame video↔ephys time lookup table |
| `EEG_EMG_ADC_from_sync.npz` | EEG/EMG/ADC channels aligned to the fit, for the next stage |
| `<video>_trimmed_to_ephys.mp4` | video trimmed to the ephys-overlapping window (optional) |

## Testing

```bash
pytest tests/
```

Unit tests run offline. A separate set of integration tests validates against
a real recording and the original MATLAB's reference output; they're gated
behind `PUP_RUN_NAS=1` since they need the raw data mounted.
