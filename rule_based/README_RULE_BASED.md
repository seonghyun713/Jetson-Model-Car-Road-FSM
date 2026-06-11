# Rule-Based FSM

This folder is the canonical home for the road-task finite state machine.

## Contents

- `scripts/rule_based_driving.py`: FSM logic used by the active Jetson runners.
- `README_RULE_BASED.md`: this note.

## Main Runtime

Run from the active Jetson runtime folder:

```bash
cd /home/ircv02/HYU-ECL3003/track_riding/model_car_jetson
./lane_following/run.sh
```

The active entrypoint is:

```text
model_car_jetson/lane_following/run.sh
```

Set `lane.lane_only_drive: false` in `model_car_jetson/lane_following/config.yaml`
when the wheel command should go through this FSM. If it is `true`, YOLO still
runs for visualization/logging but motors follow the lane controller directly.

## Key FSM File

The mission logic lives in:

```text
rule_based/scripts/rule_based_driving.py
```

It handles stop signs, pedestrian slow zones, traffic lights, roundabout route intent, and right-turn roundabout vehicle yielding.

Old duplicate Jetson runtime files from this folder were moved to:

```text
trash/20260524_rule_based_cleanup/rule_based/
```
