# Lane Following Runtime

This folder is the clean entrypoint for the model-car lane follower.

## Run

```bash
cd /home/ircv02/HYU-ECL3003/track_riding/model_car_jetson
./lane_following/run.sh
```

Open the Codex/web preview for port `8081`.

The web preview is lightweight by default:

```yaml
preview:
  enabled: true
  mode: detection
  stream_fps: 1.0
  stream_width: 640
  jpeg_quality: 55
```

`mode: detection` sends only the latest bbox image at a low rate. The browser fetches one snapshot at a time, so old frames do not pile up in the SSH tunnel. Set `enabled: false` when you want zero web-preview overhead, or `mode: dashboard` when you need the heavier lane/BEV visualization.

Stop with `Ctrl+C`.

## Tune

Edit:

```text
lane_following/config.yaml
```

Most driving experiments should only touch these sections:

- `lane`: lane estimate gains and smoothing.
- `fsm.tasks`: mission-stage and event switches for one-by-one debugging.
- `fsm.filters`: normalized detection size and bottom-y thresholds.
- `rover`: wheel speed, differential steering, pivot behavior.
- `fault_tolerance`: short lane-dropout protection and steer jump limiting.
- `hardware`: arm/serial settings.

`rover.repeat_last_command: true` decouples motor output from the vision FPS. The latest wheel command is resent at `rover.command_rate_hz` between inference frames.

`fault_tolerance.lane_lost_grace_sec` keeps the last valid lane command for a short time when only lane perception drops out. Set it to `0.0` to restore immediate lane-invalid stopping. Explicit brakes from stop signs, red lights, and roundabout yield are still immediate.

`fault_tolerance.lane_lost_speed_scale` controls how much of the last valid speed is used during that grace window. Set it to `0.0` to disable lane-lost motion; set it to `1.0` for no slowdown.

`fault_tolerance.max_steer_delta_per_frame` rejects sudden one-frame steering jumps. Set it to `0.0` to disable the limiter.

By default, `lane.lane_only_drive: false`, so the wheel command goes through the FSM below.

Set `lane.lane_only_drive: true` for lane-only hardware tests. In that mode, YOLO detections are visualized/logged, but motors follow only the lane estimate.

`lane.lane_only_drive: false` enables the canonical FSM from:

```text
/home/ircv02/HYU-ECL3003/track_riding/rule_based/scripts/rule_based_driving.py
```

The default controller is normalized pure pursuit:

```yaml
lane:
  control_mode: pure_pursuit
  pure_pursuit_gain: 0.65
  lookahead_y_ratio: 0.58
```

For this skid-steer rover, wheelbase and track width are not required for the current normalized controller. Tune `lookahead_y_ratio` first:

- smaller `lookahead_y_ratio`: farther lookahead, smoother and less reactive
- larger `lookahead_y_ratio`: closer lookahead, more reactive

Then tune `pure_pursuit_gain`:

- smaller gain: gentler turns
- larger gain: harder turns

For roundabout branch selection, tune:

- `lane.roundabout_approach_x_bias`: branch preference while entering
- `lane.roundabout_left_circulate_x_bias`: branch preference while passing the 3 o'clock exit for a left route
- `lane.roundabout_right_circulate_x_bias`: branch preference for a right route
- `lane.roundabout_left_exit_x_bias`: branch preference for the 9 o'clock exit
- `lane.roundabout_right_exit_x_bias`: branch preference for the 3 o'clock exit

Negative means BEV-left, positive means BEV-right. If the car selects the opposite branch on the real track, flip the sign of the relevant bias first.

For stop signs:

- `fsm.timing.stop_approach_sec`: keep lane following after the stop sign is confirmed
- `fsm.timing.stop_hold_sec`: brake duration after the approach delay

To print the exact underlying command:

```bash
./lane_following/run.sh --dry-run
```

## Edit Map

- Normal tuning: `model_car_jetson/lane_following/config.yaml`
- FSM state logic: `rule_based/scripts/rule_based_driving.py`
- Lane geometry/controller internals: `model_car_jetson/scripts/lane_following_core.py`
- Runtime wiring only: `model_car_jetson/lane_following/run.py`

Current mission order:

`lane follow -> RA1 roundabout -> lane follow -> side sign -> lane follow -> traffic light -> lane follow -> RA2 route sign -> RA2 roundabout -> final straight`

Key timing values:

- `fsm.timing.roundabout_to_side_sec`: lane-follow time after RA1 before the side-sign zone
- `fsm.timing.side_sign_zone_timeout_sec`: maximum side-sign zone time if no stop/pedestrian action finishes it earlier
- `fsm.timing.post_sign_lane_follow_sec`: lane-follow time between side sign and traffic light
- `fsm.timing.post_traffic_lane_follow_sec`: lane-follow time between traffic light and the RA2 route-sign zone
- `fsm.timing.traffic_clear_confirm_sec`: red absence time required after seeing a traffic light before leaving the traffic phase
- `fsm.timing.traffic_planning_window_sec`: traffic-light detections remain planning-relevant for this long after first recognition
- `fsm.timing.ra2_route_sign_timeout_sec`: warning/log threshold while waiting for the RA2 left/right sign; RA2 starts only after a sign is confirmed
- `fsm.timing.roundabout_exit_sec`: normal roundabout exit duration
- `fsm.timing.roundabout_ra2_left_exit_sec`: RA2 left-turn-only exit duration
- `fsm.timing.roundabout_ra2_right_exit_sec`: RA2 right-turn-only exit duration
- `fsm.behavior.roundabout_lane_lost_recovery_steer`: steering used when dashed lane is lost during roundabout circulation/exit; negative turns left
- `fsm.behavior.roundabout_lane_lost_recovery_speed`: low recovery speed used with that steering until the lane is seen again; set `0.0` to disable
- `fsm.behavior.lane_lost_scan_turn_speed`: in-place turn speed for non-roundabout lane-lost scan; set `0.0` to disable
- `fsm.timing.lane_lost_scan_right_sec`: right-rotation duration when randomly selected for non-roundabout lane-lost scan
- `fsm.timing.lane_lost_scan_left_sec`: left-rotation duration when randomly selected for non-roundabout lane-lost scan

Outside roundabouts, lane loss randomly selects left or right in-place scan segments using the `lane_lost_scan_*` durations, repeating until the lane is valid again. Red-light latch and explicit stop holds still brake instead of scanning.

Traffic light behavior: red stops immediately, green passes immediately, and a confirmed traffic-light object with no red for `traffic_clear_confirm_sec` also passes. After a red stop, braking is latched until green or the no-red confirmation finishes, so short red-box dropouts do not restart the car. Orange is treated as a non-stop traffic color. After first traffic-light recognition, traffic remains planning-relevant for `traffic_planning_window_sec`.

Toy-car obstacle behavior: if `toy_car` passes `obstacle_stop_conf`, `obstacle_area_min`/`obstacle_bottom_y_min`, `obstacle_height_min`, and `obstacle_x_min`/`obstacle_x_max`, the FSM brakes in any mission phase with `reason=obstacle_toy_car_global_stop`. While this stop is active, mission timers are not advanced.

## FSM Task Debug

Example: only lane + traffic light reaction, with other mission tasks disabled:

```yaml
fsm:
  map_gate_mode: off
  tasks:
    ra1_route_sign: false
    ra1_roundabout: false
    side_sign_1: false
    traffic_light: true
    side_sign_2: false
    ra2_roundabout: false
    stop_sign: false
    pedestrian_sign: false
    toy_car_yield: false
```

The detection filter units are image ratios, not meters:

- `*_min_area`: bbox area / whole image area
- `*_min_bottom_y`: bbox bottom y / image height, top `0.0`, bottom `1.0`
- `traffic_light_min_area`: whole traffic-light box area
- `traffic_color_min_area`: red/green/orange lamp box area
- `traffic_color_max_aspect_ratio`: max `max(width/height, height/width)` for red/green/orange lamp boxes; `0` disables this near-square lamp check
- `obstacle_height_min`: toy-car bbox height / image height; `0` disables this height gate
