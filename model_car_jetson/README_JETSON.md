# Model Car Jetson Deployment Package

This package is for the Jetson Orin Nano runtime smoke test.

Current scope:

- YOLOv8s object detection weights and class metadata.
- SegFormer-B0 yellow-line segmentation weights.
- Real-time lane-only runner that follows the dashed yellow centerline.
- Combined real-time runner that overlays lane segmentation, YOLO detections, and rule-based drive commands.

## Layout

```text
model_car_jetson/
  configs/
  logs/
  metadata/
  scripts/
  weights/
    detection/
    segmentation/
```

## Jetson Setup

Install the PyTorch wheel that matches your JetPack version first.
Use NVIDIA's official Jetson PyTorch installation guide for that step.
Do not install a generic x86 PyTorch wheel on Jetson.

OpenCV is usually available through JetPack. If `python3 -c "import cv2"` fails:

```bash
sudo apt update
sudo apt install -y python3-opencv
```

Then install the remaining Python packages:

```bash
cd model_car_jetson
python3 -m pip install -r requirements_jetson.txt
```

If the detection or combined runner reports that `ultralytics` is missing, install it without pulling in pip OpenCV:

```bash
python3 -m pip install --no-deps ultralytics
```

## Lane-only Real-time Test

USB camera:

```bash
python3 scripts/jetson_lane_only_runner.py --camera 0 --display --save-video --device auto
```

Shortcut:

```bash
./scripts/run_lane_only_usb.sh 0
```

CSI camera shortcut:

```bash
./scripts/run_lane_only_csi.sh 0
```

Offline video:

```bash
python3 scripts/jetson_lane_only_runner.py --video sample.mp4 --display --device auto
```

Headless run:

```bash
python3 scripts/jetson_lane_only_runner.py --camera 0 --save-video --device auto
```

The runner writes:

```text
logs/lane_only_run/lane_commands.csv
logs/lane_only_run/run_config.json
logs/lane_only_run/lane_debug.mp4
```

The command output includes:

- `state`: `LANE_FOLLOW`, `DASHED_PARTIAL`, `DASHED_RECOVERY`, `LOW_CONFIDENCE`, or `LANE_LOST`
- `steer`: normalized steering command in `[-1, 1]`
- `speed`: normalized baseline speed command

This runner only prints and logs commands. It does not control motors yet.

## BEV Tuning

The default perspective points are stored in `configs/lane_following_config.json`.
They are ratios ordered as bottom-left, bottom-right, top-right, top-left.

Override them from the command line:

```bash
python3 scripts/jetson_lane_only_runner.py \
  --camera 0 \
  --display \
  --src-points-ratio "0.08,0.95;0.92,0.95;0.64,0.43;0.36,0.43"
```

Tune this on the actual mounted camera until the dashed line appears close to vertical in the BEV panel.

## Detection Smoke Test

Camera:

```bash
python3 scripts/jetson_yolo_detection_smoke.py --camera 0 --display --device auto
```

Shortcut:

```bash
./scripts/run_detection_usb.sh 0
```

Image:

```bash
python3 scripts/jetson_yolo_detection_smoke.py --image sample.jpg --display --device auto
```

The detection runner is only a smoke test for now. Full driving should combine YOLO detections, lane commands, and `scripts/rule_based_driving.py`.

## Road-task FSM With Rover Output

The lane-drive runner uses SegFormer for yellow-line lane following and YOLO detections for a rule-based road-task FSM. The FSM can hold at stop signs, slow for pedestrian signs, wait at red/orange lights until green, remember the left/right roundabout route sign, and yield to the NPC rover on right-turn roundabout runs. It uses the same Wave Rover serial protocol as the dataset collector:

```text
{"T":1,"L":left_speed,"R":right_speed}
```

Dry-run first. This opens the camera, runs segmentation and detection, prints the wheel command, and does not write to the rover:

```bash
./scripts/run_lane_drive_csi.sh 0 --no-display --max-frames 100
```

SSH-friendly live view:

```bash
./scripts/run_lane_drive_csi.sh 0 --web --no-display
```

Forward the lane-drive dashboard from your laptop:

```bash
ssh -L 8081:127.0.0.1:8081 ircv02@<jetson-ip>
```

Open `http://127.0.0.1:8081`.

The dashboard lets you tune lane BEV/gains, YOLO confidence, FSM timing/speed/vehicle-yield thresholds, camera distance scale, manual route override, and rover wheel mixing while the runner is active. If the camera makes objects look closer than they really are, raise `Camera distance scale` above `1.0`.

If you have a local display, omit `--no-display`:

```bash
./scripts/run_lane_drive_csi.sh 0
```

Before arming, confirm the serial device:

```bash
./scripts/find_rover_serial.sh
```

On the current rover setup, the expected controller link is a Silicon Labs CP210x USB-UART bridge, usually exposed as `/dev/ttyUSB0`.

Codex/SSH preview wrapper:

```bash
./scripts/run_lane_drive_csi_preview.sh 0
```

This preview wrapper binds the dashboard to `0.0.0.0:8081` and unsets X11 display variables, which avoids the common `Authorization required` / EGL display failure path when running CSI Argus from SSH without a monitor.

Then lift the car wheels off the ground and run a very low-speed armed test:

```bash
./scripts/run_lane_drive_csi.sh 0 --no-display --arm --serial /dev/ttyUSB0 --max-wheel-speed 0.04 --max-frames 100
```

If the wheel direction is reversed, add `--invert-left`, `--invert-right`, or both. If steering is too weak or too strong, tune `--steer-mix`; if it drives too fast, lower `--max-wheel-speed`.

Ground test:

```bash
./scripts/run_lane_drive_csi.sh 0 --arm --serial /dev/ttyUSB0 --max-wheel-speed 0.06 --min-drive-confidence 0.55
```

Logs are written to:

```text
logs/lane_drive_run/lane_drive_commands.csv
logs/lane_drive_run/run_config.json
```

## Combined Lane + Detection Test

CSI camera:

```bash
./scripts/run_combined_csi.sh 0
```

USB camera:

```bash
./scripts/run_combined_usb.sh 0
```

SSH-friendly web dashboard:

```bash
./scripts/run_combined_csi_web.sh 0
```

Then forward the port from your laptop:

```bash
ssh -L 8080:127.0.0.1:8080 ircv02@<jetson-ip>
```

Open `http://127.0.0.1:8080` in your laptop browser. The dashboard streams the combined view and lets you tune BEV points, lane gains, YOLO confidence, and detection cadence while the runner is active.

Headless CSI smoke test:

```bash
./scripts/run_combined_csi.sh 0 --max-frames 30
```

The combined runner writes:

```text
logs/combined_run/combined_commands.csv
logs/combined_run/run_config.json
logs/combined_run/combined_debug.mp4
```

## CSI Camera Note

If a CSI camera is used, pass a GStreamer pipeline as `--camera`.

Example shape:

```bash
python3 scripts/jetson_lane_only_runner.py \
  --camera "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink" \
  --display \
  --device auto
```
