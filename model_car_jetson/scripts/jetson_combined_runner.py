#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs

MODEL_CAR_ROOT = Path(__file__).resolve().parents[1]
TRACK_ROOT = MODEL_CAR_ROOT.parent
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from jetson_lane_only_runner import (
    DEFAULT_MODEL_DIR,
    SegFormerYellowLinePredictor,
    get_device,
    open_capture,
    parse_points,
)
from lane_following_core import (
    BevConfig,
    LaneEstimate,
    LaneFollowerConfig,
    LaneFollowerState,
    draw_bev_debug,
    estimate_lane,
    estimate_to_dict,
    overlay_segmentation,
    perspective_matrices,
    source_points,
    warp_to_bev,
)
from rule_based.scripts.rule_based_driving import Detection, FSM_TUNABLE_FIELDS, RuleBasedDrivingFSM, detections_from_ultralytics


ROOT = MODEL_CAR_ROOT
DEFAULT_DETECTION_WEIGHTS = ROOT / "weights" / "detection" / "yolov8s_model_car_best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "combined_run"

CSV_FIELDS = [
    "frame_index",
    "timestamp_sec",
    "fps",
    "lane_valid",
    "lane_confidence",
    "lane_state",
    "lane_center_x",
    "lane_lookahead_y",
    "lane_lateral_error_px",
    "lane_heading_error_rad",
    "lane_raw_steer",
    "lane_steer",
    "lane_speed",
    "lane_width_px",
    "lane_reason",
    "lane_row_count",
    "lane_dashed_row_count",
    "drive_state",
    "drive_steer",
    "drive_speed",
    "drive_brake",
    "drive_route_intent",
    "drive_reason",
    "stable_classes",
    "detections",
]

VIS_LABELS = {
    "red_stop_sign": "stop",
    "blue_right_turn_sign": "Right",
    "blue_left_turn_sign": "Left",
    "traffic_light_red": "red",
    "traffic_light_green": "green",
    "traffic_light_orange": "orange",
    "toy_car": "car",
    "yellow_pedestrian_warning_sign": "pede",
    "traffic_light": "light",
}

CLASS_COLORS = {
    "red_stop_sign": (40, 40, 230),
    "blue_right_turn_sign": (235, 155, 45),
    "blue_left_turn_sign": (255, 95, 95),
    "traffic_light_red": (20, 20, 235),
    "traffic_light_green": (75, 210, 80),
    "traffic_light_orange": (35, 150, 245),
    "toy_car": (235, 70, 210),
    "yellow_pedestrian_warning_sign": (45, 220, 245),
    "traffic_light": (210, 210, 210),
}

LANE_PARAM_FIELDS = (
    "src_points_ratio",
    "nominal_lane_width_px",
    "base_speed",
    "min_confidence",
    "control_mode",
    "lookahead_y_ratio",
    "pure_pursuit_gain",
    "lateral_gain",
    "heading_gain",
    "steer_smoothing",
    "roundabout_route_aware",
    "roundabout_approach_x_bias",
    "roundabout_circulate_x_bias",
    "roundabout_left_circulate_x_bias",
    "roundabout_right_circulate_x_bias",
    "roundabout_exit_x_bias",
    "roundabout_left_exit_x_bias",
    "roundabout_right_exit_x_bias",
    "roundabout_route_select_strength",
    "roundabout_route_far_weight",
    "roundabout_route_center_smoothing",
)
VISION_PARAM_FIELDS = (
    "conf",
    "detect_every",
)
ROVER_PARAM_FIELDS = (
    "min_drive_confidence",
    "max_wheel_speed",
    "steer_mix",
    "speed_gain",
    "min_forward_speed",
    "turn_in_place_threshold",
    "min_inner_wheel_ratio",
    "lane_lost_grace_sec",
    "lane_lost_speed_scale",
    "lane_lost_min_speed",
    "max_steer_delta_per_frame",
    "repeat_last_command",
    "command_rate_hz",
)
FSM_BOOL_FIELDS = tuple(field_name for field_name in FSM_TUNABLE_FIELDS if field_name.startswith("enable_"))
PARAM_FIELDS = tuple(
    dict.fromkeys(
        LANE_PARAM_FIELDS
        + VISION_PARAM_FIELDS
        + FSM_TUNABLE_FIELDS
        + ROVER_PARAM_FIELDS
        + ("fsm_reset",)
    )
)


def parse_bool_value(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined lane segmentation + YOLO detection runner for Jetson.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", default="0", help="Camera index or OpenCV/GStreamer pipeline.")
    source.add_argument("--video", type=Path, default=None, help="Video file for offline real-time smoke test.")
    parser.add_argument("--segformer-model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--detection-weights", type=Path, default=DEFAULT_DETECTION_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="SegFormer torch device.")
    parser.add_argument("--half", action="store_true", help="Use FP16 for SegFormer on CUDA.")
    parser.add_argument("--yolo-device", default="auto", help="auto, cpu, cuda:0, or 0 for Ultralytics.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--detect-every", type=int, default=1, help="Run YOLO every N frames and reuse the latest boxes between runs.")
    parser.add_argument("--model-width", type=int, default=768)
    parser.add_argument("--model-height", type=int, default=432)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--frame-fps", type=float, default=30.0)
    parser.add_argument("--src-points-ratio", type=parse_points, default=None, help="BL,BR,TR,TL ratios: 'x,y;x,y;x,y;x,y'")
    parser.add_argument("--bev-width", type=int, default=640)
    parser.add_argument("--bev-height", type=int, default=720)
    parser.add_argument("--nominal-lane-width-px", type=float, default=220.0)
    parser.add_argument("--base-speed", type=float, default=0.28)
    parser.add_argument("--control-mode", default="lateral_heading", choices=["lateral_heading", "pure_pursuit"])
    parser.add_argument("--pure-pursuit-gain", type=float, default=0.75)
    parser.add_argument("--lateral-gain", type=float, default=0.85)
    parser.add_argument("--heading-gain", type=float, default=0.45)
    parser.add_argument("--steer-smoothing", type=float, default=0.62)
    parser.add_argument("--lookahead-y-ratio", type=float, default=0.64)
    parser.add_argument("--vehicle-center-x-bias", type=float, default=0.0)
    parser.add_argument("--roundabout-route-aware", type=parse_bool_value, default=True)
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
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window even if a wrapper enabled --display.")
    parser.add_argument("--no-save-video", action="store_true", help="Disable debug video even if a wrapper enabled --save-video.")
    parser.add_argument("--web", action="store_true", help="Serve an SSH-friendly MJPEG dashboard.")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--stream-fps", type=float, default=4.0, help="Max dashboard stream FPS.")
    parser.add_argument("--stream-width", type=int, default=1280, help="Resize dashboard JPEGs to this max width.")
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--threaded-capture", action="store_true", help="Continuously drain the camera and process the newest frame only.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()
    if args.no_display:
        args.display = False
    if args.no_save_video:
        args.save_video = False
    return args


def yolo_prediction_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {"imgsz": args.imgsz, "conf": args.conf, "verbose": False}
    if args.yolo_device != "auto":
        kwargs["device"] = args.yolo_device
    return kwargs


class LatestFrameCapture:
    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.ok = False
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, name="latest-frame-capture", daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                with self.lock:
                    self.ok = False
                time.sleep(0.01)
                continue
            with self.lock:
                self.ok = True
                self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            with self.lock:
                if self.ok and self.frame is not None:
                    return True, self.frame.copy()
            time.sleep(0.005)
        return False, None

    def release(self) -> None:
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


def detection_summary(detections: List[Detection]) -> str:
    if not detections:
        return "none"
    parts = []
    for det in detections:
        xyxy = [int(round(value)) for value in det.xyxy]
        parts.append(f"{det.class_name}:{det.confidence:.2f}@{xyxy}")
    return " | ".join(parts)


def detection_conf_summary(detections: List[Detection], max_items: int = 6) -> str:
    if not detections:
        return "none"
    parts = []
    for det in detections[:max_items]:
        conf = f"{det.confidence:.2f}"
        if conf.startswith("0"):
            conf = conf[1:]
        parts.append(f"{visual_label(det.class_name)}:{conf}")
    remaining = len(detections) - max_items
    if remaining > 0:
        parts.append(f"+{remaining}")
    return ",".join(parts)


SHORT_LOG_NAMES = {
    "LANE_FOLLOW": "LANE",
    "DASHED_PARTIAL": "DASH",
    "DASHED_RECOVERY": "RECOV",
    "LOW_CONFIDENCE": "LOW",
    "ROUNDABOUT_APPROACH": "RA_APP",
    "ROUNDABOUT_ENTRY_YIELD": "RA_YLD",
    "ROUNDABOUT_IN": "RA_IN",
    "ROUNDABOUT_YIELD": "RA_YLD",
    "ROUNDABOUT_EXIT": "RA_OUT",
    "STOP_SIGN_APPROACH": "STOP_A",
    "STOP_SIGN_HOLD": "STOP_H",
    "TRAFFIC_LIGHT_WAIT": "TL_WAIT",
    "RA1_ROUTE_SIGN_ZONE": "RA1_SIG",
    "RA1_ENTRY": "RA1_ENT",
    "RA1_INSIDE": "RA1_IN",
    "RA1_EXITED": "RA1_OUT",
    "SIDE_SIGN_ZONE_1": "SIGN",
    "POST_SIGN_LANE_FOLLOW": "POST_S",
    "TRAFFIC_LIGHT_ZONE": "TL",
    "POST_TRAFFIC_LANE_FOLLOW": "POST_T",
    "RA2_ROUTE_SIGN_ZONE": "RA2_SIG",
    "RA2_ENTRY": "RA2_ENT",
    "RA2_INSIDE": "RA2_IN",
    "RA2_EXITED": "RA2_OUT",
    "FINAL_STRAIGHT": "FINAL",
}


def compact_text(value: object, width: int) -> str:
    text = "-" if value is None or value == "" else str(value)
    return text.ljust(width)


def short_log_name(value: object) -> str:
    if value is None or value == "":
        return "-"
    text = str(value)
    return SHORT_LOG_NAMES.get(text, text)


def short_reason(value: object) -> str:
    text = "-" if value is None or value == "" else str(value)
    if text.startswith("roundabout_lane_lost_left_recovery_inside"):
        return "ra_lost_in"
    if text.startswith("roundabout_lane_lost_left_recovery_exit"):
        return "ra_lost_out"
    if text.startswith("lane_lost_scan_right"):
        return "scan_R"
    if text.startswith("lane_lost_scan_left"):
        return "scan_L"
    if text.startswith("roundabout_approach_"):
        return "ra_app_" + text.rsplit("_", 1)[-1][:1]
    if text.startswith("roundabout_ccw_exit_"):
        return "ra_ccw_" + text.rsplit("_", 1)[-1]
    if text.startswith("roundabout_exit_"):
        return "ra_exit_" + text.rsplit("_", 1)[-1][:1]
    if text.startswith("traffic_red_clear_wait"):
        return "red_wait"
    if text.startswith("traffic_red"):
        return "red_stop"
    if text.startswith("traffic_green"):
        return "green"
    if text.startswith("obstacle_toy_car"):
        return "car_stop"
    if text.startswith("lane_lost_hold"):
        return "lane_hold"
    if text.startswith("lane_invalid"):
        return "lane_bad"
    if text.startswith("brake:"):
        return short_reason(text[6:])
    if text.startswith("stop:"):
        return "stop"
    return short_log_name(text)


def detection_boxes(detections: List[Detection]) -> List[Dict[str, object]]:
    boxes: List[Dict[str, object]] = []
    for det in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        boxes.append(
            {
                "label": visual_label(det.class_name),
                "class_name": det.class_name,
                "confidence": round(float(det.confidence), 3),
                "x": x1,
                "y": y1,
                "w": width,
                "h": height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": width * height,
            }
        )
    return boxes


def visual_label(class_name: str) -> str:
    return VIS_LABELS.get(class_name, class_name)


def class_color(class_name: str) -> Tuple[int, int, int]:
    return CLASS_COLORS.get(class_name, (52, 211, 153))


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
        x1 = max(0, min(out.shape[1] - 1, x1))
        x2 = max(0, min(out.shape[1] - 1, x2))
        y1 = max(0, min(out.shape[0] - 1, y1))
        y2 = max(0, min(out.shape[0] - 1, y2))
        color = class_color(det.class_name)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{visual_label(det.class_name)} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        top = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(out, (x1, top), (min(out.shape[1] - 1, x1 + text_w + 8), y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def make_detection_preview_frame(
    frame: np.ndarray,
    detections: List[Detection],
    drive_command: Dict[str, object],
    fps: float,
) -> np.ndarray:
    out = draw_detections(frame, detections)
    h, w = out.shape[:2]
    bar_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, out, 0.38, 0.0, out)
    text = (
        f"fps={fps:.1f} "
        f"dr={short_log_name(drive_command.get('state'))} "
        f"ph={short_log_name(drive_command.get('mission_phase'))} "
        f"rt={drive_command.get('route_intent') or '-'} "
        f"st={float(drive_command.get('steer') or 0.0):+.2f} "
        f"sp={float(drive_command.get('speed') or 0.0):.2f} "
        f"det={detection_conf_summary(detections, max_items=4)} "
        f"why={short_reason(drive_command.get('reason'))}"
    )
    cv2.putText(
        out,
        text[:190],
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def expand_traffic_light_color_detections(frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
    expanded = list(detections)
    existing_colors = {
        det.class_name
        for det in detections
        if det.class_name in {"traffic_light_red", "traffic_light_green", "traffic_light_orange"}
    }
    if existing_colors:
        return expanded

    h, w = frame.shape[:2]
    for det in detections:
        if det.class_name != "traffic_light":
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in det.xyxy]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 + 3 or y2 <= y1 + 3:
            continue
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1] > 70
        val = hsv[:, :, 2] > 70
        hue = hsv[:, :, 0]
        active = sat & val
        masks = {
            3: active & ((hue <= 10) | (hue >= 170)),
            4: active & (hue >= 38) & (hue <= 92),
            5: active & (hue > 10) & (hue < 38),
        }
        counts = {class_id: int(mask.sum()) for class_id, mask in masks.items()}
        class_id, count = max(counts.items(), key=lambda item: item[1])
        roi_area = max(1, roi.shape[0] * roi.shape[1])
        if count < 8 or count / roi_area < 0.005:
            continue
        ys, xs = np.where(masks[class_id])
        if xs.size == 0 or ys.size == 0:
            continue
        color_x1 = x1 + int(xs.min())
        color_y1 = y1 + int(ys.min())
        color_x2 = x1 + int(xs.max()) + 1
        color_y2 = y1 + int(ys.max()) + 1
        expanded.append(
            Detection(
                class_id=class_id,
                confidence=det.confidence,
                xyxy=[float(color_x1), float(color_y1), float(color_x2), float(color_y2)],
            )
        )
    return expanded


def draw_original_panel(frame: np.ndarray, mask: np.ndarray, detections: List[Detection], bev: BevConfig) -> np.ndarray:
    out = overlay_segmentation(frame, mask)
    out = draw_detections(out, detections)
    h, w = frame.shape[:2]
    pts = source_points(w, h, bev).astype(np.int32)
    cv2.polylines(out, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)
    return out


def draw_status_bar(panel: np.ndarray, estimate: LaneEstimate, drive_command: Dict[str, object], fps: float) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    brake = "ON" if drive_command.get("brake") else "off"
    text = (
        f"fps={fps:.1f} lane={estimate.state} conf={estimate.confidence:.2f} "
        f"drive={drive_command.get('state')} steer={float(drive_command.get('steer', 0.0)):+.2f} "
        f"speed={float(drive_command.get('speed', 0.0)):.2f} brake={brake} reason={drive_command.get('reason')}"
    )
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_combined_panel(
    frame: np.ndarray,
    mask: np.ndarray,
    bev_mask: np.ndarray,
    estimate: LaneEstimate,
    bev: BevConfig,
    detections: List[Detection],
    drive_command: Dict[str, object],
    fps: float,
) -> np.ndarray:
    original = draw_original_panel(frame, mask, detections, bev)
    bev_debug = draw_bev_debug(bev_mask, estimate, bev)
    target_h = 420
    orig_w = int(round(original.shape[1] * target_h / original.shape[0]))
    original_resized = cv2.resize(original, (orig_w, target_h), interpolation=cv2.INTER_AREA)
    bev_resized = cv2.resize(bev_debug, (360, target_h), interpolation=cv2.INTER_AREA)
    panel = np.zeros((target_h, orig_w + 360, 3), dtype=np.uint8)
    panel[:, :orig_w] = original_resized
    panel[:, orig_w:] = bev_resized
    return draw_status_bar(panel, estimate, drive_command, fps)


class WebVizState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: Optional[bytes] = None
        self.status: Dict[str, object] = {}
        self.params: Dict[str, object] = {}
        self.pending_updates: List[Dict[str, str]] = []
        self.message = "ready"

    def set_frame(self, frame: bytes, status: Dict[str, object], params: Dict[str, object]) -> None:
        with self.lock:
            self.frame = frame
            self.status = status
            self.params = params

    def snapshot(self) -> Tuple[Optional[bytes], Dict[str, object], Dict[str, object], str]:
        with self.lock:
            return self.frame, dict(self.status), dict(self.params), self.message

    def queue_update(self, values: Dict[str, str]) -> None:
        with self.lock:
            self.pending_updates.append(values)
            self.message = "queued"

    def pop_updates(self) -> List[Dict[str, str]]:
        with self.lock:
            updates = self.pending_updates
            self.pending_updates = []
            return updates

    def set_message(self, message: str) -> None:
        with self.lock:
            self.message = message


def params_to_string(points: Tuple[Tuple[float, float], ...]) -> str:
    return ";".join(f"{x:.3f},{y:.3f}" for x, y in points)


def current_params(
    args: argparse.Namespace,
    bev: BevConfig,
    follower: LaneFollowerConfig,
    drive_fsm: Optional[RuleBasedDrivingFSM] = None,
) -> Dict[str, object]:
    params: Dict[str, object] = {
        "src_points_ratio": params_to_string(bev.src_points_ratio),
        "nominal_lane_width_px": follower.nominal_lane_width_px,
        "base_speed": follower.base_speed,
        "min_confidence": follower.min_confidence,
        "control_mode": follower.control_mode,
        "lookahead_y_ratio": follower.lookahead_y_ratio,
        "pure_pursuit_gain": follower.pure_pursuit_gain,
        "lateral_gain": follower.lateral_gain,
        "heading_gain": follower.heading_gain,
        "steer_smoothing": follower.steer_smoothing,
        "vehicle_center_x_bias": follower.vehicle_center_x_bias,
        "roundabout_route_aware": follower.roundabout_route_aware,
        "roundabout_approach_x_bias": follower.roundabout_approach_x_bias,
        "roundabout_circulate_x_bias": follower.roundabout_circulate_x_bias,
        "roundabout_left_circulate_x_bias": follower.roundabout_left_circulate_x_bias,
        "roundabout_right_circulate_x_bias": follower.roundabout_right_circulate_x_bias,
        "roundabout_exit_x_bias": follower.roundabout_exit_x_bias,
        "roundabout_left_exit_x_bias": follower.roundabout_left_exit_x_bias,
        "roundabout_right_exit_x_bias": follower.roundabout_right_exit_x_bias,
        "roundabout_route_select_strength": follower.roundabout_route_select_strength,
        "roundabout_route_far_weight": follower.roundabout_route_far_weight,
        "roundabout_route_center_smoothing": follower.roundabout_route_center_smoothing,
        "conf": args.conf,
        "detect_every": args.detect_every,
        "preview_mode": getattr(args, "preview_mode", "dashboard"),
        "stream_fps": getattr(args, "stream_fps", 0.0),
    }
    if drive_fsm is not None:
        params.update(drive_fsm.tunable_parameters())
    for key in ROVER_PARAM_FIELDS:
        if hasattr(args, key):
            params[key] = getattr(args, key)
    return params


def web_index() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Road FSM Live</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0b0d10; color: #f4f7fb; }
    body { margin: 0; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 390px; min-height: 100vh; }
    .stage { background: #050609; display: grid; place-items: center; padding: 10px; }
    img { width: 100%; height: auto; max-height: calc(100vh - 20px); object-fit: contain; }
    aside { border-left: 1px solid #2a3038; background: #141820; padding: 14px; overflow: auto; }
    h1 { font-size: 18px; margin: 0 0 12px; letter-spacing: 0; }
    h2 { font-size: 13px; margin: 15px 0 7px; color: #91c7ff; letter-spacing: 0; }
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
    .metric { border: 1px solid #303846; border-radius: 6px; padding: 8px; background: #0f1319; min-height: 46px; }
    .metric.wide { grid-column: 1 / -1; }
    .metric span { display: block; color: #98a5b6; font-size: 12px; }
    .metric strong { display: block; font-size: 15px; margin-top: 2px; overflow-wrap: anywhere; }
    .metric.wide strong { white-space: pre-line; line-height: 1.28; }
    form { display: block; }
    fieldset { border: 1px solid #303846; border-radius: 6px; margin: 10px 0 0; padding: 8px 10px 10px; background: #10151c; }
    legend { padding: 0 5px; color: #f0b86a; font-size: 12px; }
    label { display: block; color: #cbd5e1; font-size: 12px; }
    .control { margin-top: 9px; }
    .control.disabled { opacity: 0.42; }
    .range-row { display: grid; grid-template-columns: minmax(0, 1fr) 74px; gap: 8px; align-items: center; margin-top: 4px; }
    input, select { box-sizing: border-box; width: 100%; border: 1px solid #323b4b; border-radius: 6px; padding: 7px; background: #090d12; color: #f4f7fb; font-size: 12px; }
    input[type="range"] { padding: 0; accent-color: #66d58a; }
    input[type="number"] { text-align: right; }
    button { width: 100%; margin-top: 10px; border: 0; border-radius: 6px; padding: 10px; color: #07120b; background: #66d58a; font-weight: 700; cursor: pointer; }
    button.secondary { color: #f4f7fb; background: #2c6fbd; }
    button:disabled { opacity: 0.5; cursor: wait; }
    .message { margin-top: 10px; color: #a7b2c2; font-size: 12px; min-height: 18px; overflow-wrap: anywhere; }
    @media (max-width: 920px) { main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid #2a3038; } img { max-height: 62vh; } }
  </style>
</head>
<body>
  <main>
    <section class="stage"><img id="live" src="/frame.jpg" alt="live"></section>
    <aside>
      <h1>Road FSM Live</h1>
      <div class="metrics">
        <div class="metric"><span>FPS</span><strong id="fps">-</strong></div>
        <div class="metric"><span>Lane</span><strong id="lane_state">-</strong></div>
        <div class="metric"><span>Drive</span><strong id="drive_state">-</strong></div>
        <div class="metric"><span>Route</span><strong id="route_intent">-</strong></div>
        <div class="metric"><span>Phase</span><strong id="mission_phase">-</strong></div>
        <div class="metric"><span>Round</span><strong id="roundabout">-</strong></div>
        <div class="metric"><span>Steer</span><strong id="steer">-</strong></div>
        <div class="metric"><span>Speed</span><strong id="speed">-</strong></div>
        <div class="metric wide"><span>Reason</span><strong id="reason">-</strong></div>
        <div class="metric wide"><span>Detections</span><strong id="detections">-</strong></div>
      </div>
      <form id="params">
        <fieldset>
          <legend>Mission</legend>
          <div class="control"><label>Map gate<select name="map_gate_mode"><option value="strict">strict</option><option value="soft">soft</option><option value="off">off</option></select></label></div>
          <div class="control"><label>Route override<select name="route_override"><option value="auto">auto</option><option value="none">none</option><option value="left">left</option><option value="right">right</option><option value="clear">clear</option></select></label></div>
        </fieldset>
        <fieldset>
          <legend>Lane</legend>
          <div class="control"><label>BEV points<input name="src_points_ratio"></label></div>
          <div class="control" data-name="nominal_lane_width_px"><label>Lane width px</label><div class="range-row"><input name="nominal_lane_width_px" type="range" min="80" max="420" step="1"><input class="mirror" data-for="nominal_lane_width_px" type="number" step="1"></div></div>
          <div class="control" data-name="base_speed"><label>Base speed</label><div class="range-row"><input name="base_speed" type="range" min="0.04" max="0.40" step="0.01"><input class="mirror" data-for="base_speed" type="number" step="0.01"></div></div>
          <div class="control" data-name="min_confidence"><label>Min confidence</label><div class="range-row"><input name="min_confidence" type="range" min="0.05" max="0.90" step="0.01"><input class="mirror" data-for="min_confidence" type="number" step="0.01"></div></div>
          <div class="control"><label>Control mode<select name="control_mode"><option value="pure_pursuit">pure_pursuit</option><option value="lateral_heading">lateral_heading</option></select></label></div>
          <div class="control" data-name="lookahead_y_ratio"><label>Lookahead Y</label><div class="range-row"><input name="lookahead_y_ratio" type="range" min="0.45" max="0.85" step="0.01"><input class="mirror" data-for="lookahead_y_ratio" type="number" step="0.01"></div></div>
          <div class="control" data-name="pure_pursuit_gain"><label>Pure pursuit gain</label><div class="range-row"><input name="pure_pursuit_gain" type="range" min="0.00" max="2.00" step="0.01"><input class="mirror" data-for="pure_pursuit_gain" type="number" step="0.01"></div></div>
          <div class="control" data-name="lateral_gain"><label>Lateral gain</label><div class="range-row"><input name="lateral_gain" type="range" min="0.00" max="6.00" step="0.01"><input class="mirror" data-for="lateral_gain" type="number" step="0.01"></div></div>
          <div class="control" data-name="heading_gain"><label>Heading gain</label><div class="range-row"><input name="heading_gain" type="range" min="0.00" max="4.00" step="0.01"><input class="mirror" data-for="heading_gain" type="number" step="0.01"></div></div>
          <div class="control" data-name="steer_smoothing"><label>Steer smoothing</label><div class="range-row"><input name="steer_smoothing" type="range" min="0.00" max="0.95" step="0.01"><input class="mirror" data-for="steer_smoothing" type="number" step="0.01"></div></div>
          <div class="control" data-name="vehicle_center_x_bias"><label>Vehicle X bias</label><div class="range-row"><input name="vehicle_center_x_bias" type="range" min="-0.20" max="0.20" step="0.005"><input class="mirror" data-for="vehicle_center_x_bias" type="number" step="0.005"></div></div>
        </fieldset>
        <fieldset>
          <legend>Vision</legend>
          <div class="control" data-name="conf"><label>YOLO conf</label><div class="range-row"><input name="conf" type="range" min="0.05" max="0.90" step="0.01"><input class="mirror" data-for="conf" type="number" step="0.01"></div></div>
          <div class="control" data-name="detect_every"><label>Detect every</label><div class="range-row"><input name="detect_every" type="range" min="1" max="12" step="1"><input class="mirror" data-for="detect_every" type="number" step="1"></div></div>
        </fieldset>
        <fieldset>
          <legend>FSM</legend>
          <div class="control" data-name="detection_confirm_hits"><label>Confirm hits</label><div class="range-row"><input name="detection_confirm_hits" type="range" min="1" max="5" step="1"><input class="mirror" data-for="detection_confirm_hits" type="number" step="1"></div></div>
          <div class="control" data-name="detection_memory_hold_sec"><label>Memory hold sec</label><div class="range-row"><input name="detection_memory_hold_sec" type="range" min="0.1" max="3.0" step="0.05"><input class="mirror" data-for="detection_memory_hold_sec" type="number" step="0.05"></div></div>
          <div class="control" data-name="detection_max_misses"><label>Max misses</label><div class="range-row"><input name="detection_max_misses" type="range" min="0" max="8" step="1"><input class="mirror" data-for="detection_max_misses" type="number" step="1"></div></div>
          <div class="control" data-name="sign_conf"><label>Sign conf</label><div class="range-row"><input name="sign_conf" type="range" min="0.05" max="0.90" step="0.01"><input class="mirror" data-for="sign_conf" type="number" step="0.01"></div></div>
          <div class="control" data-name="side_sign_zone_timeout_sec"><label>Side zone timeout</label><div class="range-row"><input name="side_sign_zone_timeout_sec" type="range" min="1.0" max="12.0" step="0.1"><input class="mirror" data-for="side_sign_zone_timeout_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="intersection_cross_sec"><label>Intersection cross</label><div class="range-row"><input name="intersection_cross_sec" type="range" min="0.5" max="6.0" step="0.1"><input class="mirror" data-for="intersection_cross_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_to_side_sec"><label>Round to side sec</label><div class="range-row"><input name="roundabout_to_side_sec" type="range" min="0.0" max="4.0" step="0.1"><input class="mirror" data-for="roundabout_to_side_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="route_sign_min_area"><label>Route sign area</label><div class="range-row"><input name="route_sign_min_area" type="range" min="0.0005" max="0.080" step="0.0005"><input class="mirror" data-for="route_sign_min_area" type="number" step="0.0005"></div></div>
          <div class="control" data-name="route_sign_min_bottom_y"><label>Route sign Y</label><div class="range-row"><input name="route_sign_min_bottom_y" type="range" min="0.00" max="0.80" step="0.01"><input class="mirror" data-for="route_sign_min_bottom_y" type="number" step="0.01"></div></div>
          <div class="control" data-name="stop_sign_min_area"><label>Stop sign area</label><div class="range-row"><input name="stop_sign_min_area" type="range" min="0.0005" max="0.100" step="0.0005"><input class="mirror" data-for="stop_sign_min_area" type="number" step="0.0005"></div></div>
          <div class="control" data-name="stop_sign_min_bottom_y"><label>Stop sign Y</label><div class="range-row"><input name="stop_sign_min_bottom_y" type="range" min="0.00" max="0.80" step="0.01"><input class="mirror" data-for="stop_sign_min_bottom_y" type="number" step="0.01"></div></div>
          <div class="control" data-name="pedestrian_sign_min_area"><label>Ped sign area</label><div class="range-row"><input name="pedestrian_sign_min_area" type="range" min="0.0005" max="0.100" step="0.0005"><input class="mirror" data-for="pedestrian_sign_min_area" type="number" step="0.0005"></div></div>
          <div class="control" data-name="pedestrian_sign_min_bottom_y"><label>Ped sign Y</label><div class="range-row"><input name="pedestrian_sign_min_bottom_y" type="range" min="0.00" max="0.80" step="0.01"><input class="mirror" data-for="pedestrian_sign_min_bottom_y" type="number" step="0.01"></div></div>
          <div class="control" data-name="traffic_light_min_area"><label>Light area</label><div class="range-row"><input name="traffic_light_min_area" type="range" min="0.0001" max="0.080" step="0.0001"><input class="mirror" data-for="traffic_light_min_area" type="number" step="0.0001"></div></div>
          <div class="control" data-name="traffic_color_min_area"><label>Light color area</label><div class="range-row"><input name="traffic_color_min_area" type="range" min="0.0001" max="0.020" step="0.0001"><input class="mirror" data-for="traffic_color_min_area" type="number" step="0.0001"></div></div>
          <div class="control" data-name="traffic_color_max_aspect_ratio"><label>Light color aspect</label><div class="range-row"><input name="traffic_color_max_aspect_ratio" type="range" min="0.0" max="4.0" step="0.05"><input class="mirror" data-for="traffic_color_max_aspect_ratio" type="number" step="0.05"></div></div>
          <div class="control" data-name="traffic_light_min_bottom_y"><label>Light Y</label><div class="range-row"><input name="traffic_light_min_bottom_y" type="range" min="0.00" max="0.80" step="0.01"><input class="mirror" data-for="traffic_light_min_bottom_y" type="number" step="0.01"></div></div>
          <div class="control" data-name="stop_approach_sec"><label>Stop approach sec</label><div class="range-row"><input name="stop_approach_sec" type="range" min="0.0" max="6.0" step="0.1"><input class="mirror" data-for="stop_approach_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="stop_hold_sec"><label>Stop hold sec</label><div class="range-row"><input name="stop_hold_sec" type="range" min="0.2" max="5.0" step="0.1"><input class="mirror" data-for="stop_hold_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="pedestrian_speed_limit"><label>Pedestrian speed</label><div class="range-row"><input name="pedestrian_speed_limit" type="range" min="0.03" max="0.25" step="0.01"><input class="mirror" data-for="pedestrian_speed_limit" type="number" step="0.01"></div></div>
          <div class="control" data-name="pedestrian_hold_sec"><label>Pedestrian hold sec</label><div class="range-row"><input name="pedestrian_hold_sec" type="range" min="0.5" max="8.0" step="0.1"><input class="mirror" data-for="pedestrian_hold_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="traffic_stop_conf"><label>Traffic stop conf</label><div class="range-row"><input name="traffic_stop_conf" type="range" min="0.05" max="0.90" step="0.01"><input class="mirror" data-for="traffic_stop_conf" type="number" step="0.01"></div></div>
          <div class="control" data-name="traffic_green_conf"><label>Traffic green conf</label><div class="range-row"><input name="traffic_green_conf" type="range" min="0.05" max="0.90" step="0.01"><input class="mirror" data-for="traffic_green_conf" type="number" step="0.01"></div></div>
          <div class="control" data-name="traffic_clear_confirm_sec"><label>Traffic clear confirm</label><div class="range-row"><input name="traffic_clear_confirm_sec" type="range" min="0.0" max="3.0" step="0.1"><input class="mirror" data-for="traffic_clear_confirm_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="traffic_planning_window_sec"><label>Traffic plan window</label><div class="range-row"><input name="traffic_planning_window_sec" type="range" min="0.0" max="20.0" step="0.5"><input class="mirror" data-for="traffic_planning_window_sec" type="number" step="0.5"></div></div>
          <div class="control" data-name="roundabout_speed_limit"><label>Roundabout speed</label><div class="range-row"><input name="roundabout_speed_limit" type="range" min="0.04" max="0.25" step="0.01"><input class="mirror" data-for="roundabout_speed_limit" type="number" step="0.01"></div></div>
          <div class="control" data-name="roundabout_approach_sec"><label>Roundabout approach</label><div class="range-row"><input name="roundabout_approach_sec" type="range" min="0.2" max="4.0" step="0.1"><input class="mirror" data-for="roundabout_approach_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_entry_yield_timeout_sec"><label>Entry yield timeout</label><div class="range-row"><input name="roundabout_entry_yield_timeout_sec" type="range" min="0.5" max="8.0" step="0.1"><input class="mirror" data-for="roundabout_entry_yield_timeout_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_left_sec"><label>Roundabout left sec</label><div class="range-row"><input name="roundabout_left_sec" type="range" min="1.0" max="9.0" step="0.1"><input class="mirror" data-for="roundabout_left_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_right_sec"><label>Roundabout right sec</label><div class="range-row"><input name="roundabout_right_sec" type="range" min="1.0" max="7.0" step="0.1"><input class="mirror" data-for="roundabout_right_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_exit_sec"><label>Roundabout exit sec</label><div class="range-row"><input name="roundabout_exit_sec" type="range" min="0.2" max="4.0" step="0.1"><input class="mirror" data-for="roundabout_exit_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_lane_lost_recovery_steer"><label>Round lost steer</label><div class="range-row"><input name="roundabout_lane_lost_recovery_steer" type="range" min="-0.50" max="0.50" step="0.01"><input class="mirror" data-for="roundabout_lane_lost_recovery_steer" type="number" step="0.01"></div></div>
          <div class="control" data-name="roundabout_lane_lost_recovery_speed"><label>Round lost speed</label><div class="range-row"><input name="roundabout_lane_lost_recovery_speed" type="range" min="0.00" max="0.25" step="0.01"><input class="mirror" data-for="roundabout_lane_lost_recovery_speed" type="number" step="0.01"></div></div>
          <div class="control" data-name="lane_lost_scan_turn_speed"><label>Lane scan speed</label><div class="range-row"><input name="lane_lost_scan_turn_speed" type="range" min="0.00" max="0.30" step="0.01"><input class="mirror" data-for="lane_lost_scan_turn_speed" type="number" step="0.01"></div></div>
          <div class="control" data-name="lane_lost_scan_first_right_left_weight"><label>1st R scan L wt</label><div class="range-row"><input name="lane_lost_scan_first_right_left_weight" type="range" min="0.00" max="20.00" step="0.5"><input class="mirror" data-for="lane_lost_scan_first_right_left_weight" type="number" step="0.5"></div></div>
          <div class="control" data-name="lane_lost_scan_first_right_right_weight"><label>1st R scan R wt</label><div class="range-row"><input name="lane_lost_scan_first_right_right_weight" type="range" min="0.00" max="20.00" step="0.5"><input class="mirror" data-for="lane_lost_scan_first_right_right_weight" type="number" step="0.5"></div></div>
          <div class="control" data-name="lane_lost_scan_first_left_left_weight"><label>1st L scan L wt</label><div class="range-row"><input name="lane_lost_scan_first_left_left_weight" type="range" min="0.00" max="20.00" step="0.5"><input class="mirror" data-for="lane_lost_scan_first_left_left_weight" type="number" step="0.5"></div></div>
          <div class="control" data-name="lane_lost_scan_first_left_right_weight"><label>1st L scan R wt</label><div class="range-row"><input name="lane_lost_scan_first_left_right_weight" type="range" min="0.00" max="20.00" step="0.5"><input class="mirror" data-for="lane_lost_scan_first_left_right_weight" type="number" step="0.5"></div></div>
          <div class="control" data-name="lane_lost_scan_right_sec"><label>Lane scan right sec</label><div class="range-row"><input name="lane_lost_scan_right_sec" type="range" min="0.1" max="5.0" step="0.1"><input class="mirror" data-for="lane_lost_scan_right_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="lane_lost_scan_left_sec"><label>Lane scan left sec</label><div class="range-row"><input name="lane_lost_scan_left_sec" type="range" min="0.1" max="8.0" step="0.1"><input class="mirror" data-for="lane_lost_scan_left_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_ra2_left_exit_sec"><label>RA2 left exit sec</label><div class="range-row"><input name="roundabout_ra2_left_exit_sec" type="range" min="0.2" max="4.0" step="0.1"><input class="mirror" data-for="roundabout_ra2_left_exit_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_ra2_right_exit_sec"><label>RA2 right exit sec</label><div class="range-row"><input name="roundabout_ra2_right_exit_sec" type="range" min="0.2" max="4.0" step="0.1"><input class="mirror" data-for="roundabout_ra2_right_exit_sec" type="number" step="0.1"></div></div>
          <div class="control" data-name="roundabout_yield_clear_sec"><label>Yield clear sec</label><div class="range-row"><input name="roundabout_yield_clear_sec" type="range" min="0.2" max="2.0" step="0.05"><input class="mirror" data-for="roundabout_yield_clear_sec" type="number" step="0.05"></div></div>
          <div class="control" data-name="roundabout_left_steer_bias"><label>Left steer bias</label><div class="range-row"><input name="roundabout_left_steer_bias" type="range" min="-0.40" max="0.40" step="0.01"><input class="mirror" data-for="roundabout_left_steer_bias" type="number" step="0.01"></div></div>
          <div class="control" data-name="roundabout_right_steer_bias"><label>Right steer bias</label><div class="range-row"><input name="roundabout_right_steer_bias" type="range" min="-0.40" max="0.40" step="0.01"><input class="mirror" data-for="roundabout_right_steer_bias" type="number" step="0.01"></div></div>
          <div class="control" data-name="roundabout_ra1_circulate_bias"><label>RA1 ccw bias</label><div class="range-row"><input name="roundabout_ra1_circulate_bias" type="range" min="-0.50" max="0.50" step="0.01"><input class="mirror" data-for="roundabout_ra1_circulate_bias" type="number" step="0.01"></div></div>
          <div class="control" data-name="roundabout_ra2_circulate_bias"><label>RA2 ccw bias</label><div class="range-row"><input name="roundabout_ra2_circulate_bias" type="range" min="-0.50" max="0.50" step="0.01"><input class="mirror" data-for="roundabout_ra2_circulate_bias" type="number" step="0.01"></div></div>
          <div class="control" data-name="obstacle_stop_conf"><label>Car conf</label><div class="range-row"><input name="obstacle_stop_conf" type="range" min="0.05" max="0.90" step="0.01"><input class="mirror" data-for="obstacle_stop_conf" type="number" step="0.01"></div></div>
          <div class="control" data-name="camera_distance_scale"><label>Camera distance scale</label><div class="range-row"><input name="camera_distance_scale" type="range" min="0.50" max="2.50" step="0.01"><input class="mirror" data-for="camera_distance_scale" type="number" step="0.01"></div></div>
          <div class="control" data-name="obstacle_area_min"><label>Car area min</label><div class="range-row"><input name="obstacle_area_min" type="range" min="0.002" max="0.080" step="0.001"><input class="mirror" data-for="obstacle_area_min" type="number" step="0.001"></div></div>
          <div class="control" data-name="obstacle_height_min"><label>Car height min</label><div class="range-row"><input name="obstacle_height_min" type="range" min="0.00" max="0.80" step="0.01"><input class="mirror" data-for="obstacle_height_min" type="number" step="0.01"></div></div>
          <div class="control" data-name="obstacle_bottom_y_min"><label>Car bottom Y</label><div class="range-row"><input name="obstacle_bottom_y_min" type="range" min="0.20" max="0.95" step="0.01"><input class="mirror" data-for="obstacle_bottom_y_min" type="number" step="0.01"></div></div>
          <div class="control" data-name="obstacle_x_min"><label>Car X min</label><div class="range-row"><input name="obstacle_x_min" type="range" min="0.00" max="0.60" step="0.01"><input class="mirror" data-for="obstacle_x_min" type="number" step="0.01"></div></div>
          <div class="control" data-name="obstacle_x_max"><label>Car X max</label><div class="range-row"><input name="obstacle_x_max" type="range" min="0.40" max="1.00" step="0.01"><input class="mirror" data-for="obstacle_x_max" type="number" step="0.01"></div></div>
        </fieldset>
        <fieldset>
          <legend>Rover</legend>
          <div class="control" data-name="max_wheel_speed"><label>Max wheel speed</label><div class="range-row"><input name="max_wheel_speed" type="range" min="0.02" max="0.25" step="0.01"><input class="mirror" data-for="max_wheel_speed" type="number" step="0.01"></div></div>
          <div class="control" data-name="steer_mix"><label>Steer mix</label><div class="range-row"><input name="steer_mix" type="range" min="0.00" max="3.00" step="0.01"><input class="mirror" data-for="steer_mix" type="number" step="0.01"></div></div>
          <div class="control" data-name="speed_gain"><label>Speed gain</label><div class="range-row"><input name="speed_gain" type="range" min="0.10" max="1.80" step="0.01"><input class="mirror" data-for="speed_gain" type="number" step="0.01"></div></div>
          <div class="control" data-name="min_drive_confidence"><label>Drive min conf</label><div class="range-row"><input name="min_drive_confidence" type="range" min="0.05" max="0.90" step="0.01"><input class="mirror" data-for="min_drive_confidence" type="number" step="0.01"></div></div>
          <div class="control" data-name="min_forward_speed"><label>Min forward speed</label><div class="range-row"><input name="min_forward_speed" type="range" min="0.00" max="0.25" step="0.01"><input class="mirror" data-for="min_forward_speed" type="number" step="0.01"></div></div>
          <div class="control" data-name="turn_in_place_threshold"><label>Pivot threshold</label><div class="range-row"><input name="turn_in_place_threshold" type="range" min="0.00" max="1.00" step="0.01"><input class="mirror" data-for="turn_in_place_threshold" type="number" step="0.01"></div></div>
          <div class="control" data-name="min_inner_wheel_ratio"><label>Inner wheel min</label><div class="range-row"><input name="min_inner_wheel_ratio" type="range" min="0.00" max="0.80" step="0.01"><input class="mirror" data-for="min_inner_wheel_ratio" type="number" step="0.01"></div></div>
          <div class="control" data-name="lane_lost_grace_sec"><label>Lane lost grace</label><div class="range-row"><input name="lane_lost_grace_sec" type="range" min="0.00" max="2.00" step="0.05"><input class="mirror" data-for="lane_lost_grace_sec" type="number" step="0.05"></div></div>
          <div class="control" data-name="lane_lost_speed_scale"><label>Lost speed scale</label><div class="range-row"><input name="lane_lost_speed_scale" type="range" min="0.00" max="1.00" step="0.05"><input class="mirror" data-for="lane_lost_speed_scale" type="number" step="0.05"></div></div>
          <div class="control" data-name="lane_lost_min_speed"><label>Lost min speed</label><div class="range-row"><input name="lane_lost_min_speed" type="range" min="0.00" max="0.20" step="0.01"><input class="mirror" data-for="lane_lost_min_speed" type="number" step="0.01"></div></div>
          <div class="control" data-name="max_steer_delta_per_frame"><label>Max steer delta</label><div class="range-row"><input name="max_steer_delta_per_frame" type="range" min="0.00" max="1.00" step="0.02"><input class="mirror" data-for="max_steer_delta_per_frame" type="number" step="0.02"></div></div>
          <div class="control"><label>Repeat command<select name="repeat_last_command"><option value="true">true</option><option value="false">false</option></select></label></div>
          <div class="control" data-name="command_rate_hz"><label>Command rate</label><div class="range-row"><input name="command_rate_hz" type="range" min="1" max="50" step="1"><input class="mirror" data-for="command_rate_hz" type="number" step="1"></div></div>
        </fieldset>
        <input id="fsm_reset" name="fsm_reset" type="hidden" value="">
        <button id="apply" type="submit" disabled>Apply</button>
        <button class="secondary" id="reset" type="button" disabled>Reset FSM</button>
      </form>
      <div class="message" id="message"></div>
    </aside>
  </main>
  <script>
    const form = document.querySelector("#params");
    const liveImage = document.querySelector("#live");
    const applyButton = document.querySelector("#apply");
    const resetButton = document.querySelector("#reset");
    let touched = false;
    let loaded = false;
    function setTouched() { touched = true; }
    function syncMirror(input) {
      const mirror = form.querySelector(`.mirror[data-for="${input.name}"]`);
      if (mirror) mirror.value = input.value;
    }
    function syncFromMirror(mirror) {
      const input = form.elements[mirror.dataset.for];
      if (input) {
        input.value = mirror.value;
        syncMirror(input);
      }
    }
    form.querySelectorAll("input[name], select[name]").forEach((input) => {
      input.addEventListener("input", () => {
        setTouched();
        if (input.name) syncMirror(input);
      });
    });
    form.querySelectorAll(".mirror").forEach((mirror) => {
      mirror.addEventListener("input", () => {
        setTouched();
        syncFromMirror(mirror);
      });
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!loaded) return;
      const data = new URLSearchParams(new FormData(form));
      data.delete("fsm_reset");
      const res = await fetch("/params", {method: "POST", body: data});
      document.querySelector("#message").textContent = await res.text();
      touched = false;
    });
    resetButton.addEventListener("click", async () => {
      const data = new URLSearchParams();
      data.set("fsm_reset", "1");
      const res = await fetch("/params", {method: "POST", body: data});
      document.querySelector("#message").textContent = await res.text();
      touched = false;
    });
    function setText(id, value) { document.getElementById(id).textContent = value ?? "-"; }
    function formatDetections(status) {
      const boxes = status.detection_boxes || [];
      if (boxes.length) {
        return boxes.map((box) => {
          const conf = Number(box.confidence || 0).toFixed(2);
          return `${box.label} ${conf}  x=${box.x} y=${box.y}  w=${box.w} h=${box.h}`;
        }).join("\\n");
      }
      return (status.detections || []).join(", ") || "-";
    }
    function setField(key, value) {
      const input = form.elements[key];
      if (!input) return;
      input.value = value;
      syncMirror(input);
    }
    function setControlEnabled(key, enabled) {
      const input = form.elements[key];
      if (!input) return;
      input.disabled = !enabled;
      const control = input.closest(".control");
      if (control) control.classList.toggle("disabled", !enabled);
      const mirror = form.querySelector(`.mirror[data-for="${key}"]`);
      if (mirror) mirror.disabled = !enabled;
    }
    async function refresh() {
      try {
        const data = await (await fetch("/status.json", {cache: "no-store"})).json();
        const s = data.status || {};
        const p = data.params || {};
        setText("fps", Number(s.fps || 0).toFixed(1));
        setText("lane_state", s.lane_state);
        setText("drive_state", s.drive_state);
        setText("route_intent", s.route_intent || "-");
        setText("mission_phase", s.mission_phase || "-");
        setText("roundabout", `${s.active_roundabout_index || "-"} / ${s.roundabout_exit_index || "-"} / ${s.first_route || "-"}>${s.second_route || "-"}`);
        setText("steer", Number(s.steer || 0).toFixed(2));
        setText("speed", Number(s.speed || 0).toFixed(2));
        setText("reason", `${s.brake ? "BRAKE " : ""}${s.reason || "-"}`);
        setText("detections", formatDetections(s));
        document.querySelector("#message").textContent = data.message || "";
        const previewFps = Number(p.stream_fps || 0);
        if (previewFps > 0) frameDelayMs = Math.max(250, Math.round(1000 / previewFps));
        const present = new Set(Object.keys(p));
        form.querySelectorAll("input[name], select[name]").forEach((input) => {
          if (input.name !== "fsm_reset") setControlEnabled(input.name, present.has(input.name));
        });
        applyButton.disabled = false;
        resetButton.disabled = !present.has("route_override");
        loaded = true;
        if (!touched) {
          for (const [key, value] of Object.entries(p)) {
            setField(key, value);
          }
        }
      } catch (err) {}
    }
    let frameRefreshRunning = false;
    let frameDelayMs = 1000;
    let frameObjectUrl = "";
    async function refreshFrameOnce() {
      if (frameRefreshRunning) return;
      frameRefreshRunning = true;
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), Math.max(1200, frameDelayMs * 3));
      try {
        const res = await fetch(`/frame.jpg?t=${Date.now()}`, {cache: "no-store", signal: controller.signal});
        window.clearTimeout(timeout);
        if (res.ok && res.status !== 204) {
          const blob = await res.blob();
          const nextUrl = URL.createObjectURL(blob);
          const prevUrl = frameObjectUrl;
          frameObjectUrl = nextUrl;
          liveImage.onload = () => {
            if (prevUrl) URL.revokeObjectURL(prevUrl);
          };
          liveImage.src = nextUrl;
        }
      } catch (err) {
        window.clearTimeout(timeout);
      } finally {
        frameRefreshRunning = false;
        window.setTimeout(refreshFrameOnce, frameDelayMs);
      }
    }
    setInterval(refresh, 1000);
    refresh();
    refreshFrameOnce();
  </script>
</body>
</html>
"""


def start_web_server(state: WebVizState, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                payload = web_index()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.startswith("/status.json"):
                _, status, params, message = state.snapshot()
                payload = json.dumps({"status": status, "params": params, "message": message}).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.startswith("/frame.jpg"):
                frame, _, _, _ = state.snapshot()
                if frame is None:
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                return
            if self.path.startswith("/stream.mjpg"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while True:
                    frame, _, _, _ = state.snapshot()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    time.sleep(0.05)
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/params":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            values = {key: vals[-1] for key, vals in parse_qs(body, keep_blank_values=True).items() if key in PARAM_FIELDS}
            state.queue_update(values)
            payload = f"queued {html.escape(str(len(values)))} params".encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="web-viz", daemon=True)
    thread.start()
    return server


def encode_dashboard_frame(panel: np.ndarray, max_width: int, quality: int) -> Optional[bytes]:
    image = panel
    if max_width > 0 and panel.shape[1] > max_width:
        scale = max_width / float(panel.shape[1])
        image = cv2.resize(panel, (max_width, max(1, int(round(panel.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 35, 95))])
    if not ok:
        return None
    return encoded.tobytes()


def status_payload(
    frame_index: int,
    fps: float,
    estimate: LaneEstimate,
    drive_command: Dict[str, object],
    detections: List[Detection],
) -> Dict[str, object]:
    return {
        "frame_index": frame_index,
        "fps": fps,
        "lane_state": estimate.state,
        "lane_confidence": estimate.confidence,
        "drive_state": drive_command.get("state"),
        "steer": drive_command.get("steer"),
        "speed": drive_command.get("speed"),
        "brake": drive_command.get("brake"),
        "route_intent": drive_command.get("route_intent"),
        "mission_phase": drive_command.get("mission_phase"),
        "active_roundabout_index": drive_command.get("active_roundabout_index"),
        "first_route": drive_command.get("first_route"),
        "second_route": drive_command.get("second_route"),
        "roundabout_exit_index": drive_command.get("roundabout_exit_index"),
        "reason": drive_command.get("reason"),
        "stable_classes": drive_command.get("stable_classes", []),
        "detections": [f"{visual_label(det.class_name)} {det.confidence:.2f}" for det in detections],
        "detection_boxes": detection_boxes(detections),
    }


PARAM_RANGES = {
    "nominal_lane_width_px": (40.0, 600.0),
    "base_speed": (0.0, 0.60),
    "min_confidence": (0.0, 1.0),
    "lookahead_y_ratio": (0.10, 0.95),
    "pure_pursuit_gain": (0.0, 2.0),
    "lateral_gain": (0.0, 6.0),
    "heading_gain": (0.0, 4.0),
    "steer_smoothing": (0.0, 0.98),
    "vehicle_center_x_bias": (-0.50, 0.50),
    "roundabout_approach_x_bias": (-0.50, 0.50),
    "roundabout_circulate_x_bias": (-0.50, 0.50),
    "roundabout_left_circulate_x_bias": (-0.50, 0.50),
    "roundabout_right_circulate_x_bias": (-0.50, 0.50),
    "roundabout_exit_x_bias": (-0.50, 0.50),
    "roundabout_left_exit_x_bias": (-0.50, 0.50),
    "roundabout_right_exit_x_bias": (-0.50, 0.50),
    "roundabout_route_select_strength": (0.0, 1.0),
    "roundabout_route_far_weight": (0.0, 4.0),
    "roundabout_route_center_smoothing": (0.0, 0.98),
    "conf": (0.01, 0.99),
    "min_drive_confidence": (0.0, 1.0),
    "max_wheel_speed": (0.0, 0.50),
    "steer_mix": (0.0, 3.00),
    "speed_gain": (0.0, 2.50),
    "min_forward_speed": (0.0, 0.50),
    "turn_in_place_threshold": (0.0, 1.0),
    "min_inner_wheel_ratio": (0.0, 0.95),
    "lane_lost_grace_sec": (0.0, 5.0),
    "lane_lost_speed_scale": (0.0, 1.0),
    "lane_lost_min_speed": (0.0, 0.50),
    "max_steer_delta_per_frame": (0.0, 1.0),
    "command_rate_hz": (1.0, 50.0),
    "detection_confirm_hits": (1.0, 5.0),
    "detection_memory_hold_sec": (0.0, 5.0),
    "detection_max_misses": (0.0, 8.0),
    "route_sign_min_area": (0.0, 0.20),
    "route_sign_min_bottom_y": (0.0, 1.0),
    "stop_sign_min_area": (0.0, 0.20),
    "stop_sign_min_bottom_y": (0.0, 1.0),
    "pedestrian_sign_min_area": (0.0, 0.20),
    "pedestrian_sign_min_bottom_y": (0.0, 1.0),
    "traffic_light_min_area": (0.0, 0.20),
    "traffic_color_min_area": (0.0, 0.20),
    "traffic_color_max_aspect_ratio": (0.0, 6.0),
    "traffic_light_min_bottom_y": (0.0, 1.0),
    "side_sign_zone_timeout_sec": (0.0, 20.0),
    "intersection_cross_sec": (0.0, 10.0),
    "roundabout_to_side_sec": (0.0, 10.0),
    "stop_approach_sec": (0.0, 10.0),
    "stop_hold_sec": (0.0, 10.0),
    "stop_cooldown_sec": (0.0, 30.0),
    "sign_conf": (0.0, 1.0),
    "pedestrian_speed_limit": (0.0, 0.50),
    "pedestrian_hold_sec": (0.0, 15.0),
    "traffic_stop_conf": (0.0, 1.0),
    "traffic_green_conf": (0.0, 1.0),
    "traffic_clear_confirm_sec": (0.0, 10.0),
    "traffic_planning_window_sec": (0.0, 30.0),
    "roundabout_lane_lost_recovery_steer": (-1.0, 1.0),
    "roundabout_lane_lost_recovery_speed": (0.0, 0.50),
    "lane_lost_scan_turn_speed": (0.0, 0.60),
    "lane_lost_scan_first_right_left_weight": (0.0, 100.0),
    "lane_lost_scan_first_right_right_weight": (0.0, 100.0),
    "lane_lost_scan_first_left_left_weight": (0.0, 100.0),
    "lane_lost_scan_first_left_right_weight": (0.0, 100.0),
    "lane_lost_scan_right_sec": (0.0, 10.0),
    "lane_lost_scan_left_sec": (0.0, 10.0),
    "obstacle_stop_conf": (0.0, 1.0),
    "camera_distance_scale": (0.10, 4.0),
    "obstacle_area_min": (0.0, 0.30),
    "obstacle_height_min": (0.0, 1.0),
    "obstacle_bottom_y_min": (0.0, 1.0),
    "obstacle_x_min": (0.0, 1.0),
    "obstacle_x_max": (0.0, 1.0),
    "roundabout_speed_limit": (0.0, 0.50),
    "roundabout_approach_sec": (0.0, 10.0),
    "roundabout_entry_yield_timeout_sec": (0.0, 15.0),
    "roundabout_left_sec": (0.0, 20.0),
    "roundabout_right_sec": (0.0, 20.0),
    "roundabout_exit_sec": (0.0, 10.0),
    "roundabout_ra2_left_exit_sec": (0.0, 10.0),
    "roundabout_ra2_right_exit_sec": (0.0, 10.0),
    "roundabout_sign_cooldown_sec": (0.0, 30.0),
    "roundabout_yield_clear_sec": (0.0, 5.0),
    "roundabout_left_steer_bias": (-1.0, 1.0),
    "roundabout_right_steer_bias": (-1.0, 1.0),
    "roundabout_ra1_circulate_bias": (-1.0, 1.0),
    "roundabout_ra2_circulate_bias": (-1.0, 1.0),
    "max_steer_bias": (0.0, 1.0),
}


def bounded_param(key: str, value: float) -> float:
    low, high = PARAM_RANGES.get(key, (-float("inf"), float("inf")))
    return float(np.clip(value, low, high))


def apply_web_updates(
    updates: List[Dict[str, str]],
    args: argparse.Namespace,
    bev: BevConfig,
    follower: LaneFollowerConfig,
    matrix_cache: Dict[Tuple[int, int], np.ndarray],
    web_state: WebVizState,
    drive_fsm: Optional[RuleBasedDrivingFSM] = None,
) -> bool:
    if not updates:
        return False
    messages = []
    reset_state = False
    for update in updates:
        for key, value in update.items():
            value = value.strip()
            if not value:
                continue
            try:
                if key == "fsm_reset" and drive_fsm is not None:
                    drive_fsm.clear_route()
                    drive_fsm.stop_approach_until = 0.0
                    drive_fsm.stop_until = 0.0
                    drive_fsm.pedestrian_until = 0.0
                    drive_fsm.state = "LANE_FOLLOW"
                    messages.append("fsm_reset=1")
                    continue
                if key == "src_points_ratio":
                    bev.src_points_ratio = parse_points(value)
                    matrix_cache.clear()
                    reset_state = True
                elif key == "detect_every":
                    args.detect_every = max(1, int(float(value)))
                elif key == "conf":
                    args.conf = bounded_param(key, float(value))
                elif key == "repeat_last_command" and hasattr(args, key):
                    if value.lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                        raise ValueError("repeat_last_command must be true or false")
                    setattr(args, key, value.lower() in {"true", "1", "yes", "on"})
                elif key in ROVER_PARAM_FIELDS and hasattr(args, key):
                    setattr(args, key, bounded_param(key, float(value)))
                elif drive_fsm is not None and key in FSM_TUNABLE_FIELDS:
                    if key == "route_override":
                        if value not in {"auto", "none", "left", "right", "clear"}:
                            raise ValueError("route_override must be auto, none, left, right, or clear")
                        if value == "clear":
                            drive_fsm.clear_route()
                            drive_fsm.route_override = "auto"
                        else:
                            drive_fsm.route_override = value
                    elif key == "map_gate_mode":
                        if value not in {"strict", "soft", "off"}:
                            raise ValueError("map_gate_mode must be strict, soft, or off")
                        drive_fsm.map_gate_mode = value
                    elif key in FSM_BOOL_FIELDS:
                        setattr(drive_fsm, key, parse_bool_value(value))
                    elif key in {"detection_confirm_hits", "detection_max_misses"}:
                        setattr(drive_fsm, key, int(round(bounded_param(key, float(value)))))
                    else:
                        setattr(drive_fsm, key, bounded_param(key, float(value)))
                    if drive_fsm.obstacle_x_min > drive_fsm.obstacle_x_max:
                        drive_fsm.obstacle_x_min, drive_fsm.obstacle_x_max = (
                            drive_fsm.obstacle_x_max,
                            drive_fsm.obstacle_x_min,
                        )
                elif hasattr(follower, key):
                    if key == "control_mode":
                        if value not in {"lateral_heading", "pure_pursuit"}:
                            raise ValueError("control_mode must be lateral_heading or pure_pursuit")
                        follower.control_mode = value
                    elif key == "roundabout_route_aware":
                        follower.roundabout_route_aware = parse_bool_value(value)
                    else:
                        setattr(follower, key, bounded_param(key, float(value)))
                    reset_state = True
                else:
                    continue
                messages.append(f"{key}={value}")
            except Exception as exc:
                messages.append(f"{key} rejected: {exc}")
    if messages:
        web_state.set_message("; ".join(messages[-5:]))
    return reset_state


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    device,
    bev: BevConfig,
    follower: LaneFollowerConfig,
    drive_fsm: Optional[RuleBasedDrivingFSM] = None,
) -> None:
    payload = {
        "device": str(device),
        "yolo_device": args.yolo_device,
        "segformer_model": str(args.segformer_model),
        "detection_weights": str(args.detection_weights),
        "model_width": args.model_width,
        "model_height": args.model_height,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "detect_every": args.detect_every,
        "bev": {
            "output_width": bev.output_width,
            "output_height": bev.output_height,
            "src_points_ratio": bev.src_points_ratio,
        },
        "lane_follower": follower.__dict__,
    }
    if drive_fsm is not None:
        payload["driving_fsm"] = drive_fsm.tunable_parameters()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def csv_row(
    frame_index: int,
    started_at: float,
    fps: float,
    estimate: LaneEstimate,
    drive_command: Dict[str, object],
    detections: List[Detection],
) -> Dict[str, object]:
    lane = estimate_to_dict(estimate)
    row = {
        "frame_index": frame_index,
        "timestamp_sec": time.perf_counter() - started_at,
        "fps": fps,
        "lane_valid": lane["valid"],
        "lane_confidence": lane["confidence"],
        "lane_state": lane["state"],
        "lane_center_x": lane["center_x"],
        "lane_lookahead_y": lane["lookahead_y"],
        "lane_lateral_error_px": lane["lateral_error_px"],
        "lane_heading_error_rad": lane["heading_error_rad"],
        "lane_raw_steer": lane["raw_steer"],
        "lane_steer": lane["steer"],
        "lane_speed": lane["speed"],
        "lane_width_px": lane["lane_width_px"],
        "lane_reason": lane["reason"],
        "lane_row_count": lane["row_count"],
        "lane_dashed_row_count": lane["dashed_row_count"],
        "drive_state": drive_command.get("state"),
        "drive_steer": drive_command.get("steer"),
        "drive_speed": drive_command.get("speed"),
        "drive_brake": drive_command.get("brake"),
        "drive_route_intent": drive_command.get("route_intent"),
        "drive_reason": drive_command.get("reason"),
        "stable_classes": ",".join(drive_command.get("stable_classes", [])),
        "detections": detection_summary(detections),
    }
    return {field: row.get(field) for field in CSV_FIELDS}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.detect_every = max(1, args.detect_every)

    device = get_device(args.device)
    seg_predictor = SegFormerYellowLinePredictor(args.segformer_model, device, args.model_width, args.model_height, args.half)
    yolo_model = YOLO(str(args.detection_weights))

    bev = BevConfig(output_width=args.bev_width, output_height=args.bev_height)
    if args.src_points_ratio is not None:
        bev.src_points_ratio = args.src_points_ratio
    follower_config = LaneFollowerConfig(
        nominal_lane_width_px=args.nominal_lane_width_px,
        base_speed=args.base_speed,
        control_mode=args.control_mode,
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
    follower_state = LaneFollowerState()
    drive_fsm = RuleBasedDrivingFSM()
    web_state: Optional[WebVizState] = None
    web_server: Optional[ThreadingHTTPServer] = None
    if args.web:
        web_state = WebVizState()
        web_state.status = {"lane_state": "starting", "drive_state": "starting"}
        web_state.params = current_params(args, bev, follower_config, drive_fsm)
        web_server = start_web_server(web_state, args.web_host, args.web_port)
        print(f"web=http://{args.web_host}:{args.web_port}")

    raw_cap = open_capture(args)
    cap = LatestFrameCapture(raw_cap) if args.threaded_capture else raw_cap
    log_path = args.output_dir / "combined_commands.csv"
    config_path = args.output_dir / "run_config.json"
    video_path = args.output_dir / "combined_debug.mp4"
    write_run_config(config_path, args, device, bev, follower_config, drive_fsm)

    writer: Optional[cv2.VideoWriter] = None
    matrix_cache: Dict[Tuple[int, int], np.ndarray] = {}
    latest_detections: List[Detection] = []
    started_at = time.perf_counter()
    prev_time = started_at
    frame_index = 0
    last_stream_at = 0.0

    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            if web_state is not None:
                if apply_web_updates(web_state.pop_updates(), args, bev, follower_config, matrix_cache, web_state, drive_fsm):
                    follower_state = LaneFollowerState()

            h, w = frame.shape[:2]
            key = (w, h)
            if key not in matrix_cache:
                matrix, _ = perspective_matrices(w, h, bev)
                matrix_cache[key] = matrix

            mask = seg_predictor.predict(frame)
            bev_mask = warp_to_bev(mask, bev, matrix_cache[key], is_mask=True)
            estimate = estimate_lane(bev_mask, follower_config, follower_state, route_hint=drive_fsm.lane_route_hint())

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
            lane_command = estimate_to_dict(estimate)
            drive_command = drive_fsm.update(
                lane_command,
                latest_detections,
                now - started_at,
                frame.shape[:2],
                detections_fresh=detections_fresh,
            )
            csv_writer.writerow(csv_row(frame_index, started_at, fps, estimate, drive_command, latest_detections))

            if args.display or args.save_video or args.web:
                panel = make_combined_panel(frame, mask, bev_mask, estimate, bev, latest_detections, drive_command, fps)
                if writer is None and args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, max(1.0, args.frame_fps), (panel.shape[1], panel.shape[0]))
                if writer is not None:
                    writer.write(panel)
                if web_state is not None and now - last_stream_at >= 1.0 / max(0.1, args.stream_fps):
                    jpeg = encode_dashboard_frame(panel, args.stream_width, args.jpeg_quality)
                    if jpeg is not None:
                        web_state.set_frame(
                            jpeg,
                            status_payload(frame_index, fps, estimate, drive_command, latest_detections),
                            current_params(args, bev, follower_config, drive_fsm),
                        )
                        last_stream_at = now
                if args.display:
                    cv2.imshow("model-car combined runner", panel)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

            if args.print_every > 0 and frame_index % args.print_every == 0:
                det_conf = detection_conf_summary(latest_detections, max_items=len(latest_detections))
                print(
                    f"dr={compact_text(short_log_name(drive_command['state']), 6)} "
                    f"steer={float(drive_command['steer']):+.3f} speed={float(drive_command['speed']):.3f} "
                    f"brake={str(drive_command['brake']):<5} "
                    f"det={det_conf} "
                    f"why={drive_command.get('reason')}"
                )
            frame_index += 1

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
