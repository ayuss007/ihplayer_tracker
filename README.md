# Dynamic Player Tracker for Ice Hockey

Built to Handle Blur, Occlusions & Extreme Camera Motion.

This repository contains a production-style computer vision pipeline for **multi-object tracking in ice hockey**, designed for difficult real-game conditions:

- High-speed player motion
- Heavy motion blur
- Frequent occlusions
- Aggressive camera pan/zoom

## Features

- **RF-DETR detector integration** for players, referees, and jersey number crops.
- **PARSeq-based jersey text recognition** with confidence-weighted temporal voting.
- **Non-linear Kalman tracking** with appearance + motion association.
- **SAM + CUTIE mask propagation hooks** for silhouette-aware identity persistence.
- **ORB + ECC camera motion compensation** to stabilize track coordinates.
- **Modular architecture** for replacing models or deployment backends.

## High-level Pipeline

1. Read frame
2. Estimate global camera transform (ORB/ECC)
3. Run object detector (players/referees/jersey ROI)
4. Propagate instance masks (optional SAM/CUTIE branch)
5. OCR jersey numbers with PARSeq on number ROIs
6. Associate detections to tracks (motion + IoU + appearance)
7. Update Kalman state and identity memory
8. Emit tracks + jersey identities + diagnostics

## Repository Structure

```text
src/ih_tracker/
  config.py                 # dataclasses + runtime config
  pipeline.py               # end-to-end tracking orchestration
  detection.py              # RF-DETR wrapper
  recognition.py            # PARSeq wrapper + temporal voting
  tracking.py               # Kalman-based MOT core
  camera_motion.py          # ORB/ECC stabilization
  mask_propagation.py       # SAM/CUTIE integration points
  types.py                  # typed containers
  cli.py                    # command-line entrypoint
configs/
  default.yaml              # baseline runtime parameters
scripts/
  run_demo.py               # simple script entry
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py --video path/to/hockey.mp4 --config configs/default.yaml
```

## Resume-ready Highlights

- Designed a robust MOT pipeline specifically for fast, high-occlusion hockey scenes.
- Combined detection, sequence text recognition, and temporal voting for reliable jersey ID.
- Added camera-motion compensation for stable tracking under rapid broadcast camera movement.
- Structured the system as production-friendly modules with configuration-driven behavior.

## Notes

This starter focuses on architecture and interfaces so you can quickly plug in trained checkpoints and dataset-specific pre/post-processing.
