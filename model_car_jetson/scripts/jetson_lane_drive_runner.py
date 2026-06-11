#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MODEL_CAR_ROOT = Path(__file__).resolve().parents[1]
TRACK_ROOT = MODEL_CAR_ROOT.parent
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

import cv2
import numpy as np
import serial
from ultralytics import YOLO

from jetson_combined_runner import (
    DEFAULT_DETECTION_WEIGHTS,
    LatestFrameCapture,
    WebVizState,
    apply_web_updates,
    compact_text,
    current_params,
    detection_conf_summary,
    detection_summary,
    encode_dashboard_frame,
    expand_traffic_light_color_detections,
    make_combined_panel,
    make_detection_preview_frame,
    short_log_name,
    short_reason,
    start_web_server,
    status_payload,
    yolo_prediction_kwargs,
)
from jetson_lane_only_runner import (
    DEFAULT_MODEL_DIR,
    SegFormerYellowLinePredictor,
    get_device,
    open_capture,
    parse_points,
)
from lane_following_core import (
    BevConfig,
    LaneFollowerConfig,
    LaneFollowerState,
    estimate_lane,
    estimate_to_dict,
    perspective_matrices,
    warp_to_bev,
)
from rule_based.scripts.rule_based_driving import Detection, FSM_TUNABLE_FIELDS, RuleBasedDrivingFSM, detections_from_ultralytics


ROOT = MODEL_CAR_ROOT
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "lane_drive_run"

CSV_FIELDS = [
    "frame_index",
    "timestamp_sec",
    "fps",
    "armed",
    "lane_valid",
    "lane_confidence",
    "lane_state",
    "lane_steer",
    "lane_speed",
    "drive_state",
    "drive_steer",
    "drive_speed",
    "drive_brake",
    "drive_route_intent",
    "drive_mission_phase",
    "drive_active_roundabout_index",
    "drive_roundabout_exit_index",
    "drive_reason",
    "hardware_left",
    "hardware_right",
    "hardware_sent",
    "reason",
    "detections",
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


FSM_BOOL_FIELDS = {
    "enable_ra1_route_sign",
    "enable_ra1_roundabout",
    "enable_side_sign_1",
    "enable_traffic_light",
    "enable_side_sign_2",
    "enable_ra2_route_sign",
    "enable_ra2_roundabout",
    "enable_stop_sign",
    "enable_pedestrian_sign",
    "enable_toy_car_yield",
}
FSM_INT_FIELDS = {
    "detection_confirm_hits",
    "detection_max_misses",
}


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def option_name(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


@dataclass
class WheelCommand:
    left: float
    right: float
    sent: bool
    reason: str


class WheelCommandMixer:
    def __init__(self) -> None:
        self.last_output_steer: Optional[float] = None
        self.last_valid_steer: Optional[float] = None
        self.last_valid_speed: float = 0.0
        self.lane_lost_since: Optional[float] = None

    def command_from_drive(
        self,
        args: argparse.Namespace,
        drive_command: Dict[str, object],
        now_sec: float,
    ) -> WheelCommand:
        valid = bool(drive_command.get("lane_valid", True))
        confidence = float(drive_command.get("lane_confidence") or 0.0)
        drive_state = str(drive_command.get("state") or "")
        reason = str(drive_command.get("reason") or drive_state)
        allow_lane_fault_command = bool(drive_command.get("allow_lane_fault_command"))
        lane_fault = not valid or confidence < args.min_drive_confidence
        lane_fault_brake = bool(drive_command.get("brake")) and reason.startswith("lane_invalid")

        if bool(drive_command.get("brake")) and not lane_fault_brake:
            self._clear_lane_memory()
            return WheelCommand(0.0, 0.0, False, f"brake:{reason}")

        if lane_fault and allow_lane_fault_command and not bool(drive_command.get("brake")):
            steer = clamp(float(drive_command.get("steer") or 0.0), -1.0, 1.0)
            speed = float(drive_command.get("speed") or 0.0)
            if bool(drive_command.get("force_in_place_turn")):
                wheel = in_place_wheel_command_from_values(args, steer, speed, reason)
            else:
                wheel = wheel_command_from_values(args, steer, speed, reason)
            self.lane_lost_since = None
            self.last_output_steer = steer
            return wheel

        if lane_fault or lane_fault_brake:
            if not self._lane_fault_can_hold(drive_command):
                self._clear_lane_memory()
                if not valid:
                    return WheelCommand(0.0, 0.0, False, f"stop:{drive_state}")
                return WheelCommand(0.0, 0.0, False, f"stop:conf<{args.min_drive_confidence:.2f}")
            return self._lane_lost_hold(args, drive_command, now_sec)

        steer = clamp(float(drive_command.get("steer") or 0.0), -1.0, 1.0)
        steer = self._limit_steer_delta(args, steer)
        speed = float(drive_command.get("speed") or 0.0)
        wheel = wheel_command_from_values(args, steer, speed, reason)
        if wheel.sent:
            self.lane_lost_since = None
            self.last_valid_steer = steer
            self.last_valid_speed = speed
        else:
            self._clear_lane_memory()
        return wheel

    def _limit_steer_delta(self, args: argparse.Namespace, steer: float) -> float:
        max_delta = max(0.0, float(getattr(args, "max_steer_delta_per_frame", 0.0)))
        if max_delta > 0.0 and self.last_output_steer is not None:
            steer = clamp(steer, self.last_output_steer - max_delta, self.last_output_steer + max_delta)
        self.last_output_steer = steer
        return steer

    def _lane_fault_can_hold(self, drive_command: Dict[str, object]) -> bool:
        state = str(drive_command.get("state") or "")
        if state in {"STOP_SIGN_HOLD", "TRAFFIC_LIGHT_WAIT", "ROUNDABOUT_ENTRY_YIELD"}:
            return False
        stable = set(drive_command.get("stable_classes") or [])
        blocking_classes = {
            "traffic_light_red",
            "traffic_light_orange",
        }
        return not bool(stable & blocking_classes)

    def _lane_lost_hold(
        self,
        args: argparse.Namespace,
        drive_command: Dict[str, object],
        now_sec: float,
    ) -> WheelCommand:
        grace_sec = max(0.0, float(getattr(args, "lane_lost_grace_sec", 0.0)))
        drive_state = str(drive_command.get("state") or "")
        confidence = float(drive_command.get("lane_confidence") or 0.0)
        if grace_sec <= 0.0 or self.last_valid_steer is None or self.last_valid_speed <= 0.0:
            self._clear_lane_memory()
            if bool(drive_command.get("lane_valid", True)):
                return WheelCommand(0.0, 0.0, False, f"stop:conf<{args.min_drive_confidence:.2f}")
            return WheelCommand(0.0, 0.0, False, f"stop:{drive_state}")

        if self.lane_lost_since is None:
            self.lane_lost_since = now_sec
        elapsed = now_sec - self.lane_lost_since
        if elapsed > grace_sec:
            self._clear_lane_memory()
            if bool(drive_command.get("lane_valid", True)):
                return WheelCommand(0.0, 0.0, False, f"stop:conf<{args.min_drive_confidence:.2f}")
            return WheelCommand(0.0, 0.0, False, f"stop:{drive_state}")

        speed_scale = max(0.0, float(getattr(args, "lane_lost_speed_scale", 0.0)))
        if speed_scale <= 0.0:
            self._clear_lane_memory()
            return WheelCommand(0.0, 0.0, False, f"stop:{drive_state}")
        fallback_speed = self.last_valid_speed * speed_scale
        min_speed = max(0.0, float(getattr(args, "lane_lost_min_speed", 0.0)))
        if min_speed > 0.0:
            fallback_speed = max(fallback_speed, min_speed)
        fallback_speed = min(fallback_speed, self.last_valid_speed)
        reason = f"lane_lost_hold:{drive_state}:conf={confidence:.2f}:{elapsed:.2f}/{grace_sec:.2f}"
        return wheel_command_from_values(args, self.last_valid_steer, fallback_speed, reason)

    def _clear_lane_memory(self) -> None:
        self.last_output_steer = None
        self.last_valid_steer = None
        self.last_valid_speed = 0.0
        self.lane_lost_since = None


class RoverSerial:
    def __init__(self, serial_path: str, baud: int, armed: bool) -> None:
        self.serial_path = serial_path
        self.baud = baud
        self.armed = armed
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        if not armed:
            print("[rover] dry-run: use --arm to send motor commands.")
            return
        self._serial = serial.Serial(serial_path, baud, timeout=0.0, write_timeout=0.02)
        self._serial.reset_input_buffer()
        print(f"[rover] ARMED: connected to {serial_path} at {baud} baud")

    def send(self, left: float, right: float) -> bool:
        if not self.armed:
            return False
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                raise RuntimeError("Rover serial port is not open.")
            payload = {"T": 1, "L": round(left, 3), "R": round(right, 3)}
            self._serial.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("ascii"))
        return True

    def stop(self) -> None:
        try:
            self.send(0.0, 0.0)
        except Exception as exc:
            print(f"[rover] warning: failed to send stop: {exc}")

    def close(self) -> None:
        self.stop()
        with self._lock:
            if self._serial is not None:
                self._serial.close()


class CommandRepeater:
    def __init__(self, rover: RoverSerial, rate_hz: float) -> None:
        self.rover = rover
        self.period_sec = 1.0 / max(1.0, rate_hz)
        self._left = 0.0
        self._right = 0.0
        self._active = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rover-command-repeater", daemon=True)
        self._last_warning_at = -999.0

    def start(self) -> None:
        self._thread.start()

    def set_rate(self, rate_hz: float) -> None:
        with self._lock:
            self.period_sec = 1.0 / max(1.0, rate_hz)
        self._wake.set()

    def update(self, left: float, right: float, active: bool) -> bool:
        with self._lock:
            self._left = left if active else 0.0
            self._right = right if active else 0.0
            self._active = active
        self._wake.set()
        return self.rover.armed

    def stop(self) -> None:
        self.update(0.0, 0.0, False)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)

    def _snapshot(self) -> Tuple[float, float, bool]:
        with self._lock:
            return self._left, self._right, self._active

    def _period(self) -> float:
        with self._lock:
            return self.period_sec

    def _send_snapshot(self) -> None:
        left, right, active = self._snapshot()
        try:
            if active:
                self.rover.send(left, right)
            else:
                self.rover.send(0.0, 0.0)
        except Exception as exc:
            now = time.perf_counter()
            if now - self._last_warning_at >= 1.0:
                print(f"[rover] warning: repeated command send failed: {exc}")
                self._last_warning_at = now

    def _run(self) -> None:
        while not self._stop.is_set():
            self._send_snapshot()
            self._wake.wait(self._period())
            self._wake.clear()


def sync_command_repeater(
    args: argparse.Namespace,
    rover: RoverSerial,
    repeater: Optional[CommandRepeater],
) -> Optional[CommandRepeater]:
    args.command_rate_hz = clamp(args.command_rate_hz, 1.0, 50.0)
    if args.repeat_last_command:
        if repeater is None:
            repeater = CommandRepeater(rover, args.command_rate_hz)
            repeater.start()
            print(f"[rover] repeat-last-command enabled at {args.command_rate_hz:.1f} Hz")
        else:
            repeater.set_rate(args.command_rate_hz)
        return repeater
    if repeater is not None:
        repeater.stop()
        print("[rover] repeat-last-command disabled")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lane following runner with optional Wave Rover hardware output.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", default="0", help="Camera index or OpenCV/GStreamer pipeline.")
    source.add_argument("--video", type=Path, default=None)
    parser.add_argument("--segformer-model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--detection-weights", type=Path, default=DEFAULT_DETECTION_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--detect-every", type=int, default=5)
    parser.add_argument("--model-width", type=int, default=512)
    parser.add_argument("--model-height", type=int, default=288)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--frame-fps", type=float, default=30.0)
    parser.add_argument("--src-points-ratio", type=parse_points, default=None)
    parser.add_argument("--bev-width", type=int, default=640)
    parser.add_argument("--bev-height", type=int, default=720)
    parser.add_argument("--nominal-lane-width-px", type=float, default=220.0)
    parser.add_argument("--base-speed", type=float, default=0.20)
    parser.add_argument("--control-mode", default="lateral_heading", choices=["lateral_heading", "pure_pursuit"])
    parser.add_argument("--pure-pursuit-gain", type=float, default=0.75)
    parser.add_argument("--lateral-gain", type=float, default=0.85)
    parser.add_argument("--heading-gain", type=float, default=0.45)
    parser.add_argument("--steer-smoothing", type=float, default=0.62)
    parser.add_argument("--lookahead-y-ratio", type=float, default=0.64)
    parser.add_argument("--vehicle-center-x-bias", type=float, default=0.0)
    parser.add_argument("--roundabout-route-aware", type=parse_bool, default=True)
    parser.add_argument("--roundabout-approach-x-bias", type=float, default=0.24)
    parser.add_argument("--roundabout-circulate-x-bias", type=float, default=-0.24)
    parser.add_argument("--roundabout-left-circulate-x-bias", type=float, default=-0.24)
    parser.add_argument("--roundabout-right-circulate-x-bias", type=float, default=0.24)
    parser.add_argument("--roundabout-exit-x-bias", type=float, default=0.24)
    parser.add_argument("--roundabout-left-exit-x-bias", type=float, default=0.24)
    parser.add_argument("--roundabout-right-exit-x-bias", type=float, default=0.24)
    parser.add_argument("--roundabout-route-select-strength", type=float, default=0.75)
    parser.add_argument("--roundabout-route-far-weight", type=float, default=1.40)
    parser.add_argument("--roundabout-route-center-smoothing", type=float, default=0.30)
    parser.add_argument("--min-drive-confidence", type=float, default=0.45)
    parser.add_argument("--max-wheel-speed", type=float, default=0.10)
    parser.add_argument("--steer-mix", type=float, default=0.75)
    parser.add_argument("--speed-gain", type=float, default=1.0)
    parser.add_argument("--min-forward-speed", type=float, default=0.0)
    parser.add_argument("--turn-in-place-threshold", type=float, default=0.65)
    parser.add_argument(
        "--min-inner-wheel-ratio",
        type=float,
        default=0.0,
        help="Keep the inside wheel moving forward by at least this fraction of forward speed; 0 disables.",
    )
    parser.add_argument("--lane-lost-grace-sec", type=float, default=0.0, help="Hold the last valid lane command this long before stopping; 0 disables.")
    parser.add_argument("--lane-lost-speed-scale", type=float, default=0.55, help="Scale last valid speed during lane-lost grace; <=0 keeps the last speed.")
    parser.add_argument("--lane-lost-min-speed", type=float, default=0.0, help="Optional minimum drive speed during lane-lost grace; 0 disables.")
    parser.add_argument("--max-steer-delta-per-frame", type=float, default=0.0, help="Limit accepted steer changes per processed frame; 0 disables.")
    parser.add_argument("--repeat-last-command", action="store_true", help="Resend the latest wheel command at a fixed rate between inference frames.")
    parser.add_argument("--command-rate-hz", type=float, default=20.0, help="Wheel command resend rate when --repeat-last-command is enabled.")
    parser.add_argument("--route-override", default="auto", choices=["auto", "none", "left", "right", "clear"])
    parser.add_argument("--lane-only-drive", action="store_true", help="Use YOLO only for visualization/logging; motors follow lane output only.")
    for field_name in FSM_TUNABLE_FIELDS:
        option = option_name(field_name)
        if option in parser._option_string_actions:
            continue
        if field_name in FSM_BOOL_FIELDS:
            parser.add_argument(option, dest=field_name, type=parse_bool, default=None)
        elif field_name == "map_gate_mode":
            parser.add_argument(option, dest=field_name, default=None, choices=["strict", "soft", "off"])
        elif field_name in FSM_INT_FIELDS:
            parser.add_argument(option, dest=field_name, type=int, default=None)
        else:
            parser.add_argument(option, dest=field_name, type=float, default=None)
    parser.add_argument("--invert-left", action="store_true")
    parser.add_argument("--invert-right", action="store_true")
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--arm", action="store_true", help="Actually send wheel commands to the rover.")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window even if a wrapper enabled --display.")
    parser.add_argument("--no-save-video", action="store_true", help="Disable debug video even if a wrapper enabled --save-video.")
    parser.add_argument("--web", action="store_true", help="Serve an SSH-friendly low-rate web preview.")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8081)
    parser.add_argument("--preview-mode", choices=["detection", "dashboard"], default="detection")
    parser.add_argument("--stream-fps", type=float, default=1.0)
    parser.add_argument("--stream-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=55)
    parser.add_argument("--threaded-capture", action="store_true", help="Continuously drain the camera and process the newest frame only.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=5)
    args = parser.parse_args()
    if args.no_display:
        args.display = False
    if args.no_save_video:
        args.save_video = False
    return args


def apply_fsm_args(drive_fsm: RuleBasedDrivingFSM, args: argparse.Namespace) -> None:
    for field_name in FSM_TUNABLE_FIELDS:
        if not hasattr(args, field_name):
            continue
        value = getattr(args, field_name)
        if value is not None:
            setattr(drive_fsm, field_name, value)


def wheel_command_from_values(args: argparse.Namespace, steer: float, speed: float, reason: str) -> WheelCommand:
    steer = clamp(float(steer), -1.0, 1.0)
    forward = clamp(float(speed) * args.speed_gain, 0.0, args.max_wheel_speed)
    if forward <= 0.0:
        return WheelCommand(0.0, 0.0, False, f"stop:{reason}")
    min_forward = clamp(args.min_forward_speed, 0.0, args.max_wheel_speed)
    if min_forward > 0.0 and abs(steer) < args.turn_in_place_threshold:
        forward = max(forward, min_forward)

    # Keep enough differential torque for skid-steer turns, then optionally cap
    # it so normal cornering does not stall or reverse the inside wheel.
    mix = clamp(args.steer_mix, 0.0, 3.0)
    turn = steer * args.max_wheel_speed * mix
    pivot_enabled = args.turn_in_place_threshold < 1.0
    if pivot_enabled and abs(steer) >= args.turn_in_place_threshold:
        forward *= max(0.0, 1.0 - abs(steer))
    min_inner_ratio = clamp(getattr(args, "min_inner_wheel_ratio", 0.0), 0.0, 0.95)
    if min_inner_ratio > 0.0 and forward > 0.0:
        turn_limit = forward * (1.0 - min_inner_ratio)
        turn = clamp(turn, -turn_limit, turn_limit)

    left = forward + turn
    right = forward - turn
    left = clamp(left, -args.max_wheel_speed, args.max_wheel_speed)
    right = clamp(right, -args.max_wheel_speed, args.max_wheel_speed)
    if args.invert_left:
        left = -left
    if args.invert_right:
        right = -right
    return WheelCommand(left, right, True, reason)


def in_place_wheel_command_from_values(args: argparse.Namespace, steer: float, speed: float, reason: str) -> WheelCommand:
    steer = clamp(float(steer), -1.0, 1.0)
    turn_speed = clamp(float(speed) * args.speed_gain, 0.0, args.max_wheel_speed)
    if turn_speed <= 0.0 or abs(steer) <= 1e-6:
        return WheelCommand(0.0, 0.0, False, f"stop:{reason}")
    left = turn_speed if steer > 0.0 else -turn_speed
    right = -left
    if args.invert_left:
        left = -left
    if args.invert_right:
        right = -right
    return WheelCommand(left, right, True, reason)


def wheel_command_from_drive(args: argparse.Namespace, drive_command: Dict[str, object]) -> WheelCommand:
    return WheelCommandMixer().command_from_drive(args, drive_command, time.perf_counter())


def write_config(
    path: Path,
    args: argparse.Namespace,
    bev: BevConfig,
    follower: LaneFollowerConfig,
    drive_fsm: RuleBasedDrivingFSM,
) -> None:
    payload = {
        "armed": args.arm,
        "serial": args.serial,
        "baud": args.baud,
        "max_wheel_speed": args.max_wheel_speed,
        "steer_mix": args.steer_mix,
        "speed_gain": args.speed_gain,
        "min_forward_speed": args.min_forward_speed,
        "turn_in_place_threshold": args.turn_in_place_threshold,
        "min_inner_wheel_ratio": args.min_inner_wheel_ratio,
        "lane_lost_grace_sec": args.lane_lost_grace_sec,
        "lane_lost_speed_scale": args.lane_lost_speed_scale,
        "lane_lost_min_speed": args.lane_lost_min_speed,
        "max_steer_delta_per_frame": args.max_steer_delta_per_frame,
        "repeat_last_command": args.repeat_last_command,
        "command_rate_hz": args.command_rate_hz,
        "min_drive_confidence": args.min_drive_confidence,
        "lane_only_drive": args.lane_only_drive,
        "bev": {
            "output_width": bev.output_width,
            "output_height": bev.output_height,
            "src_points_ratio": bev.src_points_ratio,
        },
        "lane_follower": follower.__dict__,
        "driving_fsm": drive_fsm.tunable_parameters(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.detect_every = max(1, args.detect_every)

    device = get_device(args.device)
    predictor = SegFormerYellowLinePredictor(args.segformer_model, device, args.model_width, args.model_height, args.half)
    yolo_model = YOLO(str(args.detection_weights))

    bev = BevConfig(output_width=args.bev_width, output_height=args.bev_height)
    if args.src_points_ratio is not None:
        bev.src_points_ratio = args.src_points_ratio
    follower = LaneFollowerConfig(
        control_mode=args.control_mode,
        nominal_lane_width_px=args.nominal_lane_width_px,
        base_speed=args.base_speed,
        pure_pursuit_gain=args.pure_pursuit_gain,
        lateral_gain=args.lateral_gain,
        heading_gain=args.heading_gain,
        steer_smoothing=args.steer_smoothing,
        lookahead_y_ratio=args.lookahead_y_ratio,
        vehicle_center_x_bias=args.vehicle_center_x_bias,
        roundabout_route_aware=args.roundabout_route_aware,
        roundabout_approach_x_bias=args.roundabout_approach_x_bias,
        roundabout_circulate_x_bias=args.roundabout_circulate_x_bias,
        roundabout_left_circulate_x_bias=args.roundabout_left_circulate_x_bias,
        roundabout_right_circulate_x_bias=args.roundabout_right_circulate_x_bias,
        roundabout_exit_x_bias=args.roundabout_exit_x_bias,
        roundabout_left_exit_x_bias=args.roundabout_left_exit_x_bias,
        roundabout_right_exit_x_bias=args.roundabout_right_exit_x_bias,
        roundabout_route_select_strength=args.roundabout_route_select_strength,
        roundabout_route_far_weight=args.roundabout_route_far_weight,
        roundabout_route_center_smoothing=args.roundabout_route_center_smoothing,
    )
    state = LaneFollowerState()
    drive_fsm = RuleBasedDrivingFSM()
    apply_fsm_args(drive_fsm, args)
    rover = RoverSerial(args.serial, args.baud, args.arm)
    wheel_mixer = WheelCommandMixer()
    command_repeater = sync_command_repeater(args, rover, None)
    web_state: Optional[WebVizState] = None
    web_server = None
    if args.web:
        web_state = WebVizState()
        web_state.status = {"lane_state": "starting", "drive_state": "starting"}
        web_state.params = current_params(args, bev, follower, drive_fsm)
        web_server = start_web_server(web_state, args.web_host, args.web_port)
        print(f"web=http://{args.web_host}:{args.web_port}")

    raw_cap = open_capture(args)
    cap = LatestFrameCapture(raw_cap) if args.threaded_capture else raw_cap
    log_path = args.output_dir / "lane_drive_commands.csv"
    config_path = args.output_dir / "run_config.json"
    video_path = args.output_dir / "lane_drive_debug.mp4"
    write_config(config_path, args, bev, follower, drive_fsm)

    writer: Optional[cv2.VideoWriter] = None
    latest_detections: List[Detection] = []
    matrix_cache: Dict[Tuple[int, int], np.ndarray] = {}
    started_at = time.perf_counter()
    prev_time = started_at
    frame_index = 0
    last_stream_at = 0.0

    try:
        with log_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break

                if web_state is not None:
                    if apply_web_updates(web_state.pop_updates(), args, bev, follower, matrix_cache, web_state, drive_fsm):
                        state = LaneFollowerState()
                    command_repeater = sync_command_repeater(args, rover, command_repeater)

                h, w = frame.shape[:2]
                key = (w, h)
                if key not in matrix_cache:
                    matrix_cache[key], _ = perspective_matrices(w, h, bev)

                mask = predictor.predict(frame)
                bev_mask = warp_to_bev(mask, bev, matrix_cache[key], is_mask=True)
                estimate = estimate_lane(bev_mask, follower, state, route_hint=drive_fsm.lane_route_hint())

                detections_fresh = False
                if frame_index % args.detect_every == 0:
                    result = yolo_model.predict(frame, **yolo_prediction_kwargs(args))[0]
                    latest_detections = expand_traffic_light_color_detections(
                        frame,
                        detections_from_ultralytics(result),
                    )
                    detections_fresh = True

                now = time.perf_counter()
                fps = 1.0 / max(1e-6, now - prev_time)
                prev_time = now
                lane_dict = estimate_to_dict(estimate)
                if args.lane_only_drive:
                    drive_command = {
                        "state": lane_dict["state"],
                        "steer": lane_dict["steer"],
                        "speed": lane_dict["speed"],
                        "brake": False,
                        "route_intent": None,
                        "reason": "lane_only",
                        "stable_classes": [],
                        "lane_valid": lane_dict["valid"],
                        "lane_confidence": lane_dict["confidence"],
                    }
                else:
                    drive_command = drive_fsm.update(
                        lane_dict,
                        latest_detections,
                        now - started_at,
                        frame.shape[:2],
                        detections_fresh=detections_fresh,
                    )
                wheel = wheel_mixer.command_from_drive(args, drive_command, now - started_at)
                if command_repeater is not None:
                    sent = command_repeater.update(wheel.left, wheel.right, wheel.sent)
                else:
                    sent = rover.send(wheel.left, wheel.right) if wheel.sent else rover.send(0.0, 0.0)

                row = {
                    "frame_index": frame_index,
                    "timestamp_sec": now - started_at,
                    "fps": fps,
                    "armed": args.arm,
                    "lane_valid": lane_dict["valid"],
                    "lane_confidence": lane_dict["confidence"],
                    "lane_state": lane_dict["state"],
                    "lane_steer": lane_dict["steer"],
                    "lane_speed": lane_dict["speed"],
                    "drive_state": drive_command.get("state"),
                    "drive_steer": drive_command.get("steer"),
                    "drive_speed": drive_command.get("speed"),
                    "drive_brake": drive_command.get("brake"),
                    "drive_route_intent": drive_command.get("route_intent"),
                    "drive_mission_phase": drive_command.get("mission_phase"),
                    "drive_active_roundabout_index": drive_command.get("active_roundabout_index"),
                    "drive_roundabout_exit_index": drive_command.get("roundabout_exit_index"),
                    "drive_reason": drive_command.get("reason"),
                    "hardware_left": wheel.left,
                    "hardware_right": wheel.right,
                    "hardware_sent": sent,
                    "reason": wheel.reason,
                    "detections": detection_summary(latest_detections),
                }
                csv_writer.writerow({field: row.get(field) for field in CSV_FIELDS})

                if args.display or args.save_video or args.web:
                    drive_viz = dict(drive_command)
                    drive_viz["brake"] = bool(drive_command.get("brake")) or not wheel.sent or abs(wheel.left) + abs(wheel.right) < 1e-6
                    drive_viz["reason"] = f"{drive_command.get('reason')} L={wheel.left:+.2f} R={wheel.right:+.2f}"
                    stream_due = web_state is not None and now - last_stream_at >= 1.0 / max(0.1, args.stream_fps)
                    need_dashboard = args.display or args.save_video or (stream_due and args.preview_mode == "dashboard")
                    panel: Optional[np.ndarray] = None
                    if need_dashboard:
                        panel = make_combined_panel(frame, mask, bev_mask, estimate, bev, latest_detections, drive_viz, fps)
                    if panel is not None and writer is None and args.save_video:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(video_path), fourcc, max(1.0, args.frame_fps), (panel.shape[1], panel.shape[0]))
                    if panel is not None and writer is not None:
                        writer.write(panel)
                    if stream_due:
                        if args.preview_mode == "dashboard":
                            web_frame = panel if panel is not None else make_combined_panel(frame, mask, bev_mask, estimate, bev, latest_detections, drive_viz, fps)
                        else:
                            web_frame = make_detection_preview_frame(frame, latest_detections, drive_viz, fps)
                        jpeg = encode_dashboard_frame(web_frame, args.stream_width, args.jpeg_quality)
                        if jpeg is not None and web_state is not None:
                            web_state.set_frame(
                                jpeg,
                                status_payload(frame_index, fps, estimate, drive_viz, latest_detections),
                                current_params(args, bev, follower, drive_fsm),
                            )
                            last_stream_at = now
                    if args.display and panel is not None:
                        cv2.imshow("model-car lane drive runner", panel)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27, ord(" ")):
                            break

                if args.print_every > 0 and frame_index % args.print_every == 0:
                    det_conf = detection_conf_summary(latest_detections, max_items=len(latest_detections))
                    print(
                        f"dr={compact_text(short_log_name(drive_command['state']), 6)} "
                        f"ph={compact_text(short_log_name(drive_command.get('mission_phase')), 7)} "
                        f"rt={compact_text(drive_command.get('route_intent'), 5)} "
                        f"steer={float(drive_command['steer']):+.2f} speed={float(drive_command['speed']):.2f} "
                        f"det={det_conf} "
                        f"why={wheel.reason}"
                    )
                frame_index += 1
    finally:
        if command_repeater is not None:
            command_repeater.stop()
        rover.close()
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()

    print(f"frames={frame_index}")
    print(f"csv={log_path}")
    if args.save_video:
        print(f"debug_video={video_path}")


if __name__ == "__main__":
    main()
