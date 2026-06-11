#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
TRACK_ROOT = ROOT.parent
RUNNER = ROOT / "scripts" / "jetson_lane_drive_runner.py"
DEFAULT_CONFIG = APP_DIR / "config.yaml"

FSM_TASK_FLAGS = {
    "ra1_route_sign": "enable_ra1_route_sign",
    "ra1_roundabout": "enable_ra1_roundabout",
    "side_sign_1": "enable_side_sign_1",
    "traffic_light": "enable_traffic_light",
    "side_sign_2": "enable_side_sign_2",
    "ra2_route_sign": "enable_ra2_route_sign",
    "ra2_roundabout": "enable_ra2_roundabout",
    "stop_sign": "enable_stop_sign",
    "pedestrian_sign": "enable_pedestrian_sign",
    "toy_car_yield": "enable_toy_car_yield",
}

FSM_SECTIONS = {
    "detection": (
        "detection_confirm_hits",
        "detection_memory_hold_sec",
        "detection_max_misses",
        "sign_conf",
        "traffic_stop_conf",
        "traffic_green_conf",
    ),
    "filters": (
        "route_sign_min_area",
        "route_sign_min_bottom_y",
        "stop_sign_min_area",
        "stop_sign_min_bottom_y",
        "pedestrian_sign_min_area",
        "pedestrian_sign_min_bottom_y",
        "traffic_light_min_area",
        "traffic_color_min_area",
        "traffic_color_max_aspect_ratio",
        "traffic_light_min_bottom_y",
        "obstacle_stop_conf",
        "camera_distance_scale",
        "obstacle_area_min",
        "obstacle_height_min",
        "obstacle_bottom_y_min",
        "obstacle_x_min",
        "obstacle_x_max",
    ),
    "timing": (
        "side_sign_zone_timeout_sec",
        "post_sign_lane_follow_sec",
        "post_traffic_lane_follow_sec",
        "ra2_route_sign_timeout_sec",
        "intersection_cross_sec",
        "roundabout_to_side_sec",
        "stop_approach_sec",
        "stop_hold_sec",
        "stop_cooldown_sec",
        "pedestrian_hold_sec",
        "traffic_clear_confirm_sec",
        "traffic_planning_window_sec",
        "roundabout_approach_sec",
        "roundabout_entry_yield_timeout_sec",
        "roundabout_left_sec",
        "roundabout_right_sec",
        "roundabout_exit_sec",
        "roundabout_ra2_left_exit_sec",
        "roundabout_ra2_right_exit_sec",
        "lane_lost_scan_right_sec",
        "lane_lost_scan_left_sec",
        "roundabout_sign_cooldown_sec",
        "roundabout_yield_clear_sec",
    ),
    "behavior": (
        "pedestrian_speed_limit",
        "roundabout_speed_limit",
        "roundabout_approach_steer_bias",
        "roundabout_left_circulate_steer_bias",
        "roundabout_right_circulate_steer_bias",
        "roundabout_left_exit_steer_bias",
        "roundabout_right_exit_steer_bias",
        "roundabout_lane_lost_recovery_steer",
        "roundabout_lane_lost_recovery_speed",
        "lane_lost_scan_turn_speed",
        "lane_lost_scan_first_right_left_weight",
        "lane_lost_scan_first_right_right_weight",
        "lane_lost_scan_first_left_left_weight",
        "lane_lost_scan_first_left_right_weight",
        "roundabout_ra1_circulate_bias",
        "roundabout_ra2_circulate_bias",
        "roundabout_left_steer_bias",
        "roundabout_right_steer_bias",
        "max_steer_bias",
    ),
}


def flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_value(args: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    args.extend([flag(name), str(value)])


def add_bool(args: list[str], name: str, value: Any) -> None:
    if bool(value):
        args.append(flag(name))


def add_fsm_config(args: list[str], fsm: dict[str, Any]) -> None:
    fsm = fsm or {}
    add_value(args, "map_gate_mode", fsm.get("map_gate_mode"))

    tasks = fsm.get("tasks", {}) or {}
    for task_name, cli_name in FSM_TASK_FLAGS.items():
        value = tasks.get(task_name, fsm.get(cli_name))
        add_value(args, cli_name, value)

    for section_name, field_names in FSM_SECTIONS.items():
        section = fsm.get(section_name, {}) or {}
        for field_name in field_names:
            value = section.get(field_name, fsm.get(field_name))
            add_value(args, field_name, value)


def csi_pipeline(camera: dict[str, Any]) -> str:
    sensor_id = int(camera.get("sensor_id", 0))
    width = int(camera.get("width", 1280))
    height = int(camera.get("height", 720))
    fps = int(camera.get("fps", 30))
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )


def build_command(config: dict[str, Any], passthrough: list[str]) -> list[str]:
    camera = config.get("camera", {})
    preview = config.get("preview", {})
    hardware = config.get("hardware", {})
    models = config.get("models", {})
    lane = config.get("lane", {})
    rover = config.get("rover", {})
    fault_tolerance = config.get("fault_tolerance", {})
    fsm = config.get("fsm", {}) or {}
    runtime = config.get("runtime", {})

    command = [sys.executable, str(RUNNER)]

    camera_type = str(camera.get("type", "csi")).lower()
    if camera_type == "csi":
        add_value(command, "camera", csi_pipeline(camera))
    elif camera_type == "usb":
        add_value(command, "camera", camera.get("index", 0))
    elif camera_type == "video":
        add_value(command, "video", camera.get("path"))
    else:
        raise ValueError(f"Unsupported camera.type: {camera_type}")

    add_value(command, "device", models.get("device", "auto"))
    add_bool(command, "half", models.get("half", False))
    add_value(command, "yolo_device", models.get("yolo_device"))
    add_value(command, "imgsz", models.get("imgsz"))
    add_value(command, "conf", models.get("conf"))
    add_value(command, "detect_every", models.get("detect_every"))
    add_value(command, "model_width", models.get("model_width"))
    add_value(command, "model_height", models.get("model_height"))

    add_bool(command, "web", preview.get("enabled", False))
    add_value(command, "web_host", preview.get("host"))
    add_value(command, "web_port", preview.get("port"))
    if preview.get("no_display", False):
        command.append("--no-display")
    else:
        command.append("--display")
    add_value(command, "preview_mode", preview.get("mode"))
    add_value(command, "stream_fps", preview.get("stream_fps"))
    add_value(command, "stream_width", preview.get("stream_width"))
    add_value(command, "jpeg_quality", preview.get("jpeg_quality"))

    add_value(command, "serial", hardware.get("serial"))
    add_value(command, "baud", hardware.get("baud"))
    add_bool(command, "arm", hardware.get("arm", False))

    add_bool(command, "lane_only_drive", lane.get("lane_only_drive", False))
    add_value(command, "route_override", lane.get("route_override"))
    add_value(command, "nominal_lane_width_px", lane.get("nominal_lane_width_px"))
    add_value(command, "base_speed", lane.get("base_speed"))
    add_value(command, "control_mode", lane.get("control_mode"))
    add_value(command, "pure_pursuit_gain", lane.get("pure_pursuit_gain"))
    add_value(command, "lateral_gain", lane.get("lateral_gain"))
    add_value(command, "heading_gain", lane.get("heading_gain"))
    add_value(command, "steer_smoothing", lane.get("steer_smoothing"))
    add_value(command, "lookahead_y_ratio", lane.get("lookahead_y_ratio"))
    add_value(command, "vehicle_center_x_bias", lane.get("vehicle_center_x_bias"))
    add_value(command, "roundabout_route_aware", lane.get("roundabout_route_aware"))
    add_value(command, "roundabout_approach_x_bias", lane.get("roundabout_approach_x_bias"))
    add_value(command, "roundabout_circulate_x_bias", lane.get("roundabout_circulate_x_bias"))
    add_value(command, "roundabout_left_circulate_x_bias", lane.get("roundabout_left_circulate_x_bias"))
    add_value(command, "roundabout_right_circulate_x_bias", lane.get("roundabout_right_circulate_x_bias"))
    add_value(command, "roundabout_exit_x_bias", lane.get("roundabout_exit_x_bias"))
    add_value(command, "roundabout_left_exit_x_bias", lane.get("roundabout_left_exit_x_bias"))
    add_value(command, "roundabout_right_exit_x_bias", lane.get("roundabout_right_exit_x_bias"))
    add_value(command, "roundabout_route_select_strength", lane.get("roundabout_route_select_strength"))
    add_value(command, "roundabout_route_far_weight", lane.get("roundabout_route_far_weight"))
    add_value(command, "roundabout_route_center_smoothing", lane.get("roundabout_route_center_smoothing"))
    add_value(command, "min_drive_confidence", lane.get("min_drive_confidence"))
    add_fsm_config(command, fsm)

    add_value(command, "max_wheel_speed", rover.get("max_wheel_speed"))
    add_value(command, "speed_gain", rover.get("speed_gain"))
    add_value(command, "steer_mix", rover.get("steer_mix"))
    add_value(command, "min_forward_speed", rover.get("min_forward_speed"))
    add_value(command, "turn_in_place_threshold", rover.get("turn_in_place_threshold"))
    add_value(command, "min_inner_wheel_ratio", rover.get("min_inner_wheel_ratio"))
    add_value(command, "lane_lost_grace_sec", fault_tolerance.get("lane_lost_grace_sec"))
    add_value(command, "lane_lost_speed_scale", fault_tolerance.get("lane_lost_speed_scale"))
    add_value(command, "lane_lost_min_speed", fault_tolerance.get("lane_lost_min_speed"))
    add_value(command, "max_steer_delta_per_frame", fault_tolerance.get("max_steer_delta_per_frame"))
    add_bool(command, "repeat_last_command", rover.get("repeat_last_command", False))
    add_value(command, "command_rate_hz", rover.get("command_rate_hz"))
    add_bool(command, "invert_left", rover.get("invert_left", False))
    add_bool(command, "invert_right", rover.get("invert_right", False))

    add_bool(command, "threaded_capture", runtime.get("threaded_capture", True))
    add_value(command, "print_every", runtime.get("print_every"))
    add_value(command, "max_frames", runtime.get("max_frames"))
    add_bool(command, "save_video", runtime.get("save_video", False))
    if not runtime.get("save_video", False):
        command.append("--no-save-video")

    command.extend(passthrough)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the model-car lane follower from a YAML config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    parser.add_argument("--sensor-id", type=int, default=None, help="Override camera.sensor_id from the config.")
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--arm", dest="arm_override", action="store_true", default=None, help="Force hardware arm on.")
    arm_group.add_argument("--no-arm", dest="arm_override", action="store_false", help="Force hardware arm off.")
    args, passthrough = parser.parse_known_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.sensor_id is not None:
        config.setdefault("camera", {})["sensor_id"] = args.sensor_id
    if args.arm_override is not None:
        config.setdefault("hardware", {})["arm"] = args.arm_override

    command = build_command(config, passthrough)
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(TRACK_ROOT)]
    if pythonpath:
        pythonpath_parts.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if config.get("preview", {}).get("no_display", False):
        env["MODEL_CAR_NO_X11"] = "1"
        env.pop("DISPLAY", None)
        env.pop("XAUTHORITY", None)

    print("[lane_following] command:")
    print(" ".join(subprocess.list2cmdline([part]) for part in command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
