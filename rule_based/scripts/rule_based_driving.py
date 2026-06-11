#!/usr/bin/env python3
# 로직 요약:
# 조건:
# 1. confidence: 신뢰도, 크기, 위치 필터 후 confirmed: 확정 처리.
# 2. strict: 엄격 모드는 현재 zone: 구간에 맞는 이벤트만 반응.
# 3. RA1_ROUTE_SIGN_ZONE: 1차 회전교차로 경로, 좌/우 표지판으로 방향 결정.
# 4. SIDE_SIGN_ZONE_1: 표지판, 정지 표지판은 정지, 보행자는 감속.
# 5. POST_SIGN_LANE_FOLLOW: 표지판 이후 신호등까지 차선 추종.
# 6. TRAFFIC_LIGHT_ZONE: 신호등, 빨강 정지, 없으면 확인 후 통과.
# 7. toy_car: 크기/위치 필터를 통과하면 어느 구간이든 우선 정지.
# 8. lane_invalid: 기본 정지. 단, 회전교차로/일반 차선 유실 recovery에서는 저속 탐색.
# zone 순서:
# 1. RA1_ROUTE_SIGN_ZONE: 1차 회전교차로
# 2. RA1_ENTRY: 1차 회전교차로 진입
# 3. RA1_INSIDE: 1차 회전교차로 내부
# 4. RA1_EXITED: 1차 회전교차로 탈출
# 5. SIDE_SIGN_ZONE_1: 첫 번째 표지판
# 6. POST_SIGN_LANE_FOLLOW: 표지판 후 신호등까지 차선 추종
# 7. TRAFFIC_LIGHT_ZONE: 신호등, 빨강 정지, 일정 시간 빨강이 없으면 통과
# 8. POST_TRAFFIC_LANE_FOLLOW: 신호등 후 2차 경로 표지판까지 차선 추종
# 9. RA2_ROUTE_SIGN_ZONE: 2차 회전교차로 경로, 좌/우 표지판으로 방향 결정
# 10. RA2_ENTRY: 2차 회전교차로 진입
# 11. RA2_INSIDE: 2차 회전교차로 내부
# 12. RA2_EXITED: 2차 회전교차로 탈출
# 13. FINAL_STRAIGHT: 마지막 직선
# 14. INTERSECTION_CROSS/SIDE_SIGN_ZONE_2: 이전 코스용 legacy phase
# 15. soft/off: zone 제한 완화
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple


CLASS_NAMES = {
    0: "red_stop_sign",
    1: "blue_right_turn_sign",
    2: "blue_left_turn_sign",
    3: "traffic_light_red",
    4: "traffic_light_green",
    5: "traffic_light_orange",
    6: "toy_car",
    7: "yellow_pedestrian_warning_sign",
    8: "traffic_light",
}


# 튜닝 가능한 FSM 파라미터 목록입니다.
FSM_TUNABLE_FIELDS = (
    "map_gate_mode",
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
    "detection_confirm_hits",
    "detection_memory_hold_sec",
    "detection_max_misses",
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
    "side_sign_zone_timeout_sec",
    "post_sign_lane_follow_sec",
    "post_traffic_lane_follow_sec",
    "ra2_route_sign_timeout_sec",
    "intersection_cross_sec",
    "roundabout_to_side_sec",
    "stop_approach_sec",
    "stop_hold_sec",
    "stop_cooldown_sec",
    "sign_conf",
    "pedestrian_speed_limit",
    "pedestrian_hold_sec",
    "traffic_stop_conf",
    "traffic_green_conf",
    "traffic_clear_confirm_sec",
    "traffic_planning_window_sec",
    "obstacle_stop_conf",
    "camera_distance_scale",
    "obstacle_area_min",
    "obstacle_height_min",
    "obstacle_bottom_y_min",
    "obstacle_x_min",
    "obstacle_x_max",
    "roundabout_speed_limit",
    "roundabout_approach_sec",
    "roundabout_entry_yield_timeout_sec",
    "roundabout_left_sec",
    "roundabout_right_sec",
    "roundabout_exit_sec",
    "roundabout_ra2_left_exit_sec",
    "roundabout_ra2_right_exit_sec",
    "lane_lost_scan_right_sec",
    "lane_lost_scan_left_sec",
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
    "roundabout_sign_cooldown_sec",
    "roundabout_yield_clear_sec",
    "roundabout_left_steer_bias",
    "roundabout_right_steer_bias",
    "max_steer_bias",
    "route_override",
)

ROUNDABOUT_STATES = {
    "ROUNDABOUT_APPROACH",
    "ROUNDABOUT_ENTRY_YIELD",
    "ROUNDABOUT_IN",
    "ROUNDABOUT_YIELD",
    "ROUNDABOUT_EXIT",
}

MISSION_PHASES = {
    "RA1_ROUTE_SIGN_ZONE",
    "RA1_ENTRY",
    "RA1_INSIDE",
    "RA1_EXITED",
    "SIDE_SIGN_ZONE_1",
    "POST_SIGN_LANE_FOLLOW",
    "TRAFFIC_LIGHT_ZONE",
    "POST_TRAFFIC_LANE_FOLLOW",
    "RA2_ROUTE_SIGN_ZONE",
    "INTERSECTION_CROSS",
    "SIDE_SIGN_ZONE_2",
    "RA2_ENTRY",
    "RA2_INSIDE",
    "RA2_EXITED",
    "FINAL_STRAIGHT",
}


@dataclass
class Detection:
    class_id: int
    confidence: float
    xyxy: List[float]

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))


@dataclass
class DetectionSnapshot:
    class_name: str
    confidence: float
    xyxy: List[float]
    area_ratio: float
    bottom_y_ratio: float
    center_x_ratio: float
    adjusted_area_ratio: float
    adjusted_bottom_y_ratio: float


@dataclass
class DetectionMemory:
    hits: int = 0
    misses: int = 0
    last_seen_sec: float = -999.0
    consumed_until_sec: float = -999.0
    snapshot: Optional[DetectionSnapshot] = None

    def update(
        self,
        snapshot: Optional[DetectionSnapshot],
        now_sec: float,
        max_misses: int,
        max_hits: int,
    ) -> None:
        if snapshot is not None:
            self.snapshot = snapshot
            self.last_seen_sec = now_sec
            self.hits = min(max_hits, self.hits + 1)
            self.misses = 0
            return
        self.misses += 1
        if self.misses > max_misses:
            self.hits = max(0, self.hits - 1)

    def confirmed(self, now_sec: float, min_hits: int, hold_sec: float) -> bool:
        if self.snapshot is None:
            return False
        if now_sec < self.consumed_until_sec:
            return False
        if self.hits < min_hits:
            return False
        return now_sec - self.last_seen_sec <= hold_sec

    def consume(self, now_sec: float, cooldown_sec: float) -> None:
        self.consumed_until_sec = now_sec + cooldown_sec
        self.hits = 0
        self.misses = 0


@dataclass
class TemporalClassFilter:
    window: int = 5
    min_hits: int = 2
    history: Deque[set] = field(default_factory=deque)

    def update(self, detections: Iterable[Detection], min_confidence: float = 0.30) -> set:
        present = {det.class_name for det in detections if det.confidence >= min_confidence}
        self.history.append(present)
        while len(self.history) > self.window:
            self.history.popleft()
        stable = set()
        names = set().union(*self.history) if self.history else set()
        for name in names:
            if sum(1 for frame_names in self.history if name in frame_names) >= self.min_hits:
                stable.add(name)
        return stable


@dataclass
class RuleBasedDrivingFSM:
    # 맵 기반 감지 허용 범위
    map_gate_mode: str = "strict"                      # strict/soft/off

    # task별 ON/OFF. YAML에서 하나씩 끄고 켜며 디버깅합니다.
    enable_ra1_route_sign: bool = True                 # 1차 경로 표지판 감지
    enable_ra1_roundabout: bool = True                 # 1차 회전교차로 FSM
    enable_side_sign_1: bool = True                    # 첫 번째 측면 표지판 구간
    enable_traffic_light: bool = True                  # 신호등 구간
    enable_side_sign_2: bool = False                   # legacy 두 번째 측면 표지판 구간
    enable_ra2_route_sign: bool = True                 # 2차 경로 표지판 재인식
    enable_ra2_roundabout: bool = True                 # 2차 회전교차로 FSM
    enable_stop_sign: bool = True                      # 정지 표지판 반응
    enable_pedestrian_sign: bool = True                # 보행자 표지판 감속
    enable_toy_car_yield: bool = True                  # 회전교차로 toy_car 양보

    # 감지 흔들림 보정값
    detection_confirm_hits: int = 1                    # 확정 필요 횟수
    detection_memory_hold_sec: float = 2.0             # 확정 유지 시간
    detection_max_misses: int = 5                      # 미감지 허용 횟수

    # 경로 표지판 필터
    route_sign_min_area: float = 0.005                 # 최소 면적
    route_sign_min_bottom_y: float = 0.08              # 최소 하단 위치

    # 정지/보행자 표지판 필터
    stop_sign_min_area: float = 0.0030                 # 최소 면적
    stop_sign_min_bottom_y: float = 0.00               # 최소 하단 위치
    pedestrian_sign_min_area: float = 0.0030           # 최소 면적
    pedestrian_sign_min_bottom_y: float = 0.00         # 최소 하단 위치

    # 신호등 필터
    traffic_light_min_area: float = 0.004           # 최소 면적
    traffic_color_min_area: float = 0.0005          # 빨강/초록/주황 램프 최소 면적
    traffic_color_max_aspect_ratio: float = 1.8     # 램프 bbox 최대 가로/세로 비율. 0이면 비활성
    traffic_light_min_bottom_y: float = 0.02           # 최소 하단 위치

    # 미션 구간 전환 타이머
    side_sign_zone_timeout_sec: float = 5.5            # 표지판 구간 시간
    post_sign_lane_follow_sec: float = 2.2             # 표지판 후 신호등까지 추종 시간
    post_traffic_lane_follow_sec: float = 8.0          # 신호등 후 2차 경로 표지판까지 추종 시간
    ra2_route_sign_timeout_sec: float = 3.0            # 2차 표지판 미감지 경고 시간
    intersection_cross_sec: float = 2.2                # 이전 코스용 교차로 통과 시간
    roundabout_to_side_sec: float = 0.4                # 회전교차로 후 전환 시간

    # 정지 표지판 동작
    stop_approach_sec: float = 2.0                     # 감지 후 lane follow 접근 시간
    stop_hold_sec: float = 2.0                         # 정지 유지 시간
    stop_cooldown_sec: float = 6.0                     # 재정지 방지 시간

    # 표지판 confidence 기준
    sign_conf: float = 0.25                            # 표지판 신뢰도

    # 보행자 경고 감속값
    pedestrian_speed_limit: float = 0.12               # 감속 속도
    pedestrian_hold_sec: float = 4.0                   # 감속 유지 시간

    # 신호등 confidence 기준
    traffic_stop_conf: float = 0.25                    # 빨강/주황 신뢰도
    traffic_green_conf: float = 0.25                   # 초록 신뢰도
    traffic_clear_confirm_sec: float = 1.0             # 빨강 없음 확인 시간
    traffic_planning_window_sec: float = 10.0          # 첫 인식 후 신호등 반영 시간

    # 회전교차로 양보용 toy_car 필터
    obstacle_stop_conf: float = 0.30                   # 차량 신뢰도
    camera_distance_scale: float = 1.0                 # 거리감 보정
    obstacle_area_min: float = 0.018                   # 최소 면적
    obstacle_height_min: float = 0.0                   # 최소 bbox 높이 비율. 0이면 비활성
    obstacle_bottom_y_min: float = 0.54                # 최소 하단 위치
    obstacle_x_min: float = 0.15                       # x 최소 범위
    obstacle_x_max: float = 0.85                       # x 최대 범위

    # 회전교차로 속도/진입/회전/탈출 시간
    roundabout_speed_limit: float = 0.13               # 속도 제한
    roundabout_approach_sec: float = 1.8               # 진입 전 접근 시간
    roundabout_entry_yield_timeout_sec: float = 4.0    # 양보 최대 대기 시간
    roundabout_left_sec: float = 7.2                   # 좌회전 회전 시간
    roundabout_right_sec: float = 3.6                  # 우회전 회전 시간
    roundabout_exit_sec: float = 1.2                   # 탈출 시간
    roundabout_ra2_left_exit_sec: float = 1.2          # 2차 좌회전 탈출 시간
    roundabout_ra2_right_exit_sec: float = 0.7         # 2차 우회전 탈출 시간
    lane_lost_scan_right_sec: float = 1.0              # 일반 차선 유실 시 오른쪽 제자리 회전 시간
    lane_lost_scan_left_sec: float = 2.0               # 일반 차선 유실 시 왼쪽 제자리 회전 시간

    # 회전교차로 phase별 조향 보정값
    roundabout_approach_steer_bias: float = 0.15       # 진입 시 +방향 보정
    roundabout_left_circulate_steer_bias: float = -0.18 # 좌회전 순환 시 -방향 보정
    roundabout_right_circulate_steer_bias: float = 0.18 # 우회전 시 +방향 보정
    roundabout_left_exit_steer_bias: float = 0.12      # 좌회전 탈출 시 +방향 보정
    roundabout_right_exit_steer_bias: float = 0.18     # 우회전 탈출 시 +방향 보정
    roundabout_lane_lost_recovery_steer: float = -0.12 # 회전교차로 차선 유실 시 왼쪽 탐색 조향
    roundabout_lane_lost_recovery_speed: float = 0.08  # 회전교차로 차선 유실 시 탐색 속도
    lane_lost_scan_turn_speed: float = 0.08            # 일반 차선 유실 시 제자리 회전 속도. 0이면 비활성
    lane_lost_scan_first_right_left_weight: float = 1.0 # 1차 우회전 후 일반 차선 유실 시 왼쪽 탐색 가중치
    lane_lost_scan_first_right_right_weight: float = 1.0 # 1차 우회전 후 일반 차선 유실 시 오른쪽 탐색 가중치
    lane_lost_scan_first_left_left_weight: float = 1.0  # 1차 좌회전 후 일반 차선 유실 시 왼쪽 탐색 가중치
    lane_lost_scan_first_left_right_weight: float = 1.0 # 1차 좌회전 후 일반 차선 유실 시 오른쪽 탐색 가중치
    # 이전 이름입니다. 새 phase별 bias를 사용하므로 기본 제어에는 더하지 않습니다.
    roundabout_ra1_circulate_bias: float = 0.0
    roundabout_ra2_circulate_bias: float = 0.0
    roundabout_sign_cooldown_sec: float = 8.0          # 표지판 재반응 방지
    roundabout_yield_clear_sec: float = 0.65           # 차량 없음 확인 시간
    roundabout_left_steer_bias: float = 0.0            # 이전 이름
    roundabout_right_steer_bias: float = 0.0           # 이전 이름
    max_steer_bias: float = 0.35                       # 보정 최대값

    # 테스트용 강제 경로
    route_override: str = "auto"                       # auto/left/right/clear/none

    # 아래는 실행 중 상태값
    state: str = "LANE_FOLLOW"
    mission_phase: str = "RA1_ROUTE_SIGN_ZONE"
    mission_phase_started_at: float = 0.0
    active_roundabout_index: int = 0
    first_route: Optional[str] = None
    second_route: Optional[str] = None
    route_intent: Optional[str] = None
    roundabout_exit_index: int = 0
    stop_approach_until: float = 0.0
    stop_until: float = 0.0
    last_stop_sign_time: float = -999.0
    pedestrian_until: float = 0.0
    last_traffic_stop_time: float = -999.0
    traffic_first_seen_at: Optional[float] = None
    traffic_clear_since: Optional[float] = None
    traffic_red_latched: bool = False
    traffic_passed: bool = False
    roundabout_phase_started_at: float = 0.0
    last_roundabout_sign_time: float = -999.0
    vehicle_clear_since: Optional[float] = None
    roundabout_resume_state: str = "ROUNDABOUT_IN"
    lane_lost_recovery_started_at: Optional[float] = None
    lane_lost_scan_direction: int = 0
    class_filter: TemporalClassFilter = field(default_factory=TemporalClassFilter)
    detection_memory: Dict[str, DetectionMemory] = field(default_factory=dict)

    def update(
        self,
        lane_command: Dict[str, float],
        detections: Iterable[Detection],
        now_sec: float,
        frame_shape: Optional[Tuple[int, int]] = None,
        detections_fresh: bool = True,
    ) -> Dict[str, object]:
        detections = list(detections)
        self._update_detection_memory(detections, now_sec, frame_shape, detections_fresh)
        stable = self._confirmed_classes(now_sec)

        steer = float(lane_command.get("steer", 0.0))
        speed = float(lane_command.get("speed", 0.0))
        lane_state = str(lane_command.get("state", "LANE_FOLLOW"))
        lane_valid = bool(lane_command.get("valid", True))
        lane_confidence = float(lane_command.get("confidence") or 0.0)
        reason = lane_state

        if self._confirmed_vehicle_hazard(now_sec):
            self._reset_lane_lost_recovery()
            return self._command(
                steer=steer,
                speed=0.0,
                brake=True,
                reason="obstacle_toy_car_global_stop",
                stable=stable,
                lane_valid=lane_valid,
                lane_confidence=lane_confidence,
            )

        self._advance_timed_mission(now_sec)
        self._skip_disabled_mission_tasks(now_sec)
        self._capture_route_intent(now_sec)
        if (
            self.enable_pedestrian_sign
            and self._map_allows("side_sign")
            and self._confirmed("yellow_pedestrian_warning_sign", now_sec)
        ):
            self.pedestrian_until = max(self.pedestrian_until, now_sec + self.pedestrian_hold_sec)
            self._consume("yellow_pedestrian_warning_sign", now_sec, self.pedestrian_hold_sec + 2.0)

        if not lane_valid and self._roundabout_lane_lost_recovery_enabled():
            return self._roundabout_lane_lost_recovery_command(
                lane_state=lane_state,
                stable=stable,
                lane_valid=lane_valid,
                lane_confidence=lane_confidence,
                now_sec=now_sec,
            )

        if not lane_valid and self._lane_lost_scan_recovery_enabled(stable):
            return self._lane_lost_scan_recovery_command(
                lane_state=lane_state,
                stable=stable,
                lane_valid=lane_valid,
                lane_confidence=lane_confidence,
                now_sec=now_sec,
            )

        if not lane_valid:
            self._reset_lane_lost_recovery()
            return self._command(
                steer=steer,
                speed=0.0,
                brake=True,
                reason=f"lane_invalid:{lane_state}",
                stable=stable,
                lane_valid=lane_valid,
                lane_confidence=lane_confidence,
            )

        self._reset_lane_lost_recovery()

        if self.state in ROUNDABOUT_STATES and not self._roundabout_task_enabled(self.active_roundabout_index):
            self.state = "LANE_FOLLOW"
            self.route_intent = None
            self.roundabout_exit_index = 0
            self.active_roundabout_index = 0
            self.vehicle_clear_since = None
            reason = "roundabout_disabled"

        if self.state in ROUNDABOUT_STATES:
            return self._update_roundabout_state(steer, speed, stable, lane_valid, lane_confidence, now_sec)

        traffic_allowed_by_map = self.enable_traffic_light and self._map_allows("traffic_light")
        traffic_gate = self.enable_traffic_light and (
            traffic_allowed_by_map or self._traffic_planning_window_active(now_sec)
        )
        red_light = traffic_gate and self._confirmed("traffic_light_red", now_sec)
        orange_light = traffic_gate and self._confirmed("traffic_light_orange", now_sec)
        green = traffic_gate and self._confirmed("traffic_light_green", now_sec)
        traffic_light_seen = traffic_gate and self._confirmed("traffic_light", now_sec)
        traffic_present = traffic_light_seen or green or red_light or orange_light
        if traffic_allowed_by_map and traffic_present:
            self._remember_traffic_first_seen(now_sec)
        if self.state == "TRAFFIC_LIGHT_WAIT":
            self.state = "LANE_FOLLOW"
        latest_red = self._last_seen("traffic_light_red")
        green_allows_go = green and self._last_seen("traffic_light_green") > latest_red
        if traffic_gate and red_light and not green_allows_go:
            self._set_phase("TRAFFIC_LIGHT_ZONE", now_sec, reset_timer=False)
            self.traffic_clear_since = None
            self.traffic_red_latched = True
            self.traffic_passed = False
            self.last_traffic_stop_time = now_sec
            return self._command(
                steer=steer,
                speed=0.0,
                brake=True,
                reason="traffic_red",
                stable=stable,
                lane_valid=lane_valid,
                lane_confidence=lane_confidence,
            )
        if traffic_gate and green_allows_go:
            self._finish_traffic_phase(now_sec)
            reason = "traffic_green"
        elif traffic_gate and self.traffic_red_latched and not self.traffic_passed:
            self._set_phase("TRAFFIC_LIGHT_ZONE", now_sec, reset_timer=False)
            confirm_sec = max(0.0, self.traffic_clear_confirm_sec)
            if self.traffic_clear_since is None:
                self.traffic_clear_since = now_sec
            clear_elapsed = now_sec - self.traffic_clear_since
            if clear_elapsed >= confirm_sec:
                self._finish_traffic_phase(now_sec)
                reason = "traffic_red_clear_confirmed"
            else:
                return self._command(
                    steer=steer,
                    speed=0.0,
                    brake=True,
                    reason=f"traffic_red_clear_wait:{clear_elapsed:.2f}/{confirm_sec:.2f}",
                    stable=stable,
                    lane_valid=lane_valid,
                    lane_confidence=lane_confidence,
                )
        elif traffic_gate and traffic_present and not self.traffic_passed:
            self._set_phase("TRAFFIC_LIGHT_ZONE", now_sec, reset_timer=False)
            confirm_sec = max(0.0, self.traffic_clear_confirm_sec)
            if self.traffic_clear_since is None:
                self.traffic_clear_since = now_sec
            clear_elapsed = now_sec - self.traffic_clear_since
            if clear_elapsed >= confirm_sec:
                self._finish_traffic_phase(now_sec)
                reason = "traffic_clear_confirmed"
            else:
                reason = f"traffic_clear_confirming:{clear_elapsed:.2f}/{confirm_sec:.2f}"
        elif not traffic_present:
            self.traffic_clear_since = None

        stop_seen = (
            self.enable_stop_sign
            and self._map_allows("side_sign")
            and self._confirmed("red_stop_sign", now_sec)
        )
        if stop_seen and now_sec - self.last_stop_sign_time > self.stop_cooldown_sec:
            self.state = "STOP_SIGN_APPROACH"
            self.stop_approach_until = now_sec + max(0.0, self.stop_approach_sec)
            self.stop_until = 0.0
            self.last_stop_sign_time = now_sec
            self._consume("red_stop_sign", now_sec, self.stop_cooldown_sec)
        if self.state == "STOP_SIGN_APPROACH":
            if now_sec < self.stop_approach_until:
                return self._command(
                    steer=steer,
                    speed=speed,
                    brake=False,
                    reason="stop_sign_approach",
                    stable=stable,
                    lane_valid=lane_valid,
                    lane_confidence=lane_confidence,
                )
            self.state = "STOP_SIGN_HOLD"
            self.stop_until = now_sec + self.stop_hold_sec
        if self.state == "STOP_SIGN_HOLD":
            if now_sec < self.stop_until:
                return self._command(
                    steer=steer,
                    speed=0.0,
                    brake=True,
                    reason="stop_sign_hold",
                    stable=stable,
                    lane_valid=lane_valid,
                    lane_confidence=lane_confidence,
                )
            self.state = "LANE_FOLLOW"
            self._finish_side_sign_zone(now_sec)
            reason = "stop_sign_release"

        if now_sec < self.pedestrian_until:
            self.state = "PEDESTRIAN_SLOW"
            speed = min(speed, self.pedestrian_speed_limit)
            reason = "pedestrian_warning_slow"
        else:
            if self.state == "PEDESTRIAN_SLOW":
                self._finish_side_sign_zone(now_sec)
            self.state = "LANE_FOLLOW"
        if self.mission_phase == "RA2_ROUTE_SIGN_ZONE" and self.state == "LANE_FOLLOW" and reason == lane_state:
            elapsed = now_sec - self.mission_phase_started_at
            if self.ra2_route_sign_timeout_sec > 0.0 and elapsed >= self.ra2_route_sign_timeout_sec:
                reason = "ra2_route_sign_wait_timeout"
            else:
                reason = "ra2_route_sign_search"

        return self._command(
            steer=steer,
            speed=speed,
            brake=False,
            reason=reason,
            stable=stable,
            lane_valid=lane_valid,
            lane_confidence=lane_confidence,
        )

    def tunable_parameters(self) -> Dict[str, object]:
        return {field_name: getattr(self, field_name) for field_name in FSM_TUNABLE_FIELDS}

    def lane_route_hint(self) -> str:
        if self.state not in ROUNDABOUT_STATES:
            return "none"
        if self.route_intent not in {"left", "right"}:
            return "none"
        if self.state in {"ROUNDABOUT_APPROACH", "ROUNDABOUT_ENTRY_YIELD"}:
            return "approach"
        if self.state == "ROUNDABOUT_IN":
            return "right_circulate" if self.route_intent == "right" else "left_circulate"
        if self.state == "ROUNDABOUT_EXIT":
            return "left_exit" if self.route_intent == "left" else "right_exit"
        return "none"

    def clear_route(self) -> None:
        self.route_intent = None
        self.roundabout_exit_index = 0
        self.active_roundabout_index = 0
        self.first_route = None
        self.second_route = None
        if self.state in ROUNDABOUT_STATES or self.state in {"PEDESTRIAN_SLOW", "STOP_SIGN_APPROACH", "STOP_SIGN_HOLD"}:
            self.state = "LANE_FOLLOW"
        self._set_phase("RA1_ROUTE_SIGN_ZONE", 0.0)
        self.roundabout_phase_started_at = 0.0
        self.stop_approach_until = 0.0
        self.stop_until = 0.0
        self.traffic_first_seen_at = None
        self.traffic_clear_since = None
        self.traffic_red_latched = False
        self.traffic_passed = False
        self.vehicle_clear_since = None
        self._reset_lane_lost_recovery()
        self.detection_memory.clear()

    def _command(
        self,
        steer: float,
        speed: float,
        brake: bool,
        reason: str,
        stable: set,
        lane_valid: bool,
        lane_confidence: float,
        allow_lane_fault_command: bool = False,
        force_in_place_turn: bool = False,
    ) -> Dict[str, object]:
        return {
            "state": self.state,
            "steer": max(-1.0, min(1.0, steer)),
            "speed": 0.0 if brake else max(0.0, speed),
            "brake": brake,
            "route_intent": self.route_intent,
            "mission_phase": self.mission_phase,
            "active_roundabout_index": self.active_roundabout_index,
            "first_route": self.first_route,
            "second_route": self.second_route,
            "roundabout_exit_index": self.roundabout_exit_index,
            "reason": reason,
            "stable_classes": sorted(stable),
            "lane_valid": lane_valid,
            "lane_confidence": lane_confidence,
            "allow_lane_fault_command": allow_lane_fault_command,
            "force_in_place_turn": force_in_place_turn,
        }

    def _update_roundabout_state(
        self,
        steer: float,
        speed: float,
        stable: set,
        lane_valid: bool,
        lane_confidence: float,
        now_sec: float,
    ) -> Dict[str, object]:
        if self.state == "ROUNDABOUT_APPROACH":
            self._set_phase(f"RA{self.active_roundabout_index}_ENTRY", now_sec, reset_timer=False)
            speed = min(speed, self.roundabout_speed_limit)
            steer = self._biased_roundabout_phase_steer(steer)
            reason = f"roundabout_approach_{self.route_intent or 'unknown'}"
            if now_sec - self.roundabout_phase_started_at >= self.roundabout_approach_sec:
                if self.route_intent == "right":
                    self.state = "ROUNDABOUT_ENTRY_YIELD"
                else:
                    self.state = "ROUNDABOUT_IN"
                    self._set_phase(f"RA{self.active_roundabout_index}_INSIDE", now_sec)
                self.roundabout_phase_started_at = now_sec
        elif self.state == "ROUNDABOUT_ENTRY_YIELD":
            self._set_phase(f"RA{self.active_roundabout_index}_ENTRY", now_sec, reset_timer=False)
            speed = min(speed, self.roundabout_speed_limit)
            hazard = self._confirmed_vehicle_hazard(now_sec)
            timed_out = now_sec - self.roundabout_phase_started_at >= self.roundabout_entry_yield_timeout_sec
            if hazard and not timed_out:
                self.vehicle_clear_since = None
                return self._command(
                    steer=steer,
                    speed=0.0,
                    brake=True,
                    reason="roundabout_entry_yield_vehicle",
                    stable=stable,
                    lane_valid=lane_valid,
                    lane_confidence=lane_confidence,
                )
            if hazard:
                reason = "roundabout_entry_yield_timeout"
            elif self._vehicle_clear(now_sec):
                reason = "roundabout_entry_clear"
            else:
                return self._command(
                    steer=steer,
                    speed=0.0,
                    brake=True,
                    reason="roundabout_entry_clear_confirm",
                    stable=stable,
                    lane_valid=lane_valid,
                    lane_confidence=lane_confidence,
                )
            self.state = "ROUNDABOUT_IN"
            self._set_phase(f"RA{self.active_roundabout_index}_INSIDE", now_sec)
            self.roundabout_phase_started_at = now_sec
        elif self.state == "ROUNDABOUT_IN":
            self._set_phase(f"RA{self.active_roundabout_index}_INSIDE", now_sec, reset_timer=False)
            speed = min(speed, self.roundabout_speed_limit)
            steer = self._biased_roundabout_phase_steer(steer)
            duration = self.roundabout_right_sec if self.roundabout_exit_index == 1 else self.roundabout_left_sec
            reason = f"roundabout_ccw_exit_{self.roundabout_exit_index or 'unknown'}"
            if now_sec - self.roundabout_phase_started_at >= duration:
                self.state = "ROUNDABOUT_EXIT"
                self.roundabout_phase_started_at = now_sec
        elif self.state == "ROUNDABOUT_EXIT":
            self._set_phase(f"RA{self.active_roundabout_index}_INSIDE", now_sec, reset_timer=False)
            speed = min(speed, max(self.roundabout_speed_limit, self.pedestrian_speed_limit))
            steer = self._biased_roundabout_phase_steer(steer)
            reason = f"roundabout_exit_{self.route_intent or 'auto'}"
            duration = self._roundabout_exit_duration()
            if now_sec - self.roundabout_phase_started_at >= duration:
                finished_index = self.active_roundabout_index
                self.state = "LANE_FOLLOW"
                self.route_intent = None
                self.roundabout_exit_index = 0
                self.active_roundabout_index = 0
                next_phase = "RA1_EXITED" if finished_index == 1 else "RA2_EXITED"
                self._set_phase(next_phase, now_sec)
                reason = "roundabout_done"
        else:
            reason = "roundabout_state_unknown"

        return self._command(
            steer=steer,
            speed=speed,
            brake=False,
            reason=reason,
            stable=stable,
            lane_valid=lane_valid,
            lane_confidence=lane_confidence,
        )

    def _capture_route_intent(self, now_sec: float) -> None:
        detected_intent: Optional[str] = None
        if not (self.enable_ra1_roundabout or self.enable_ra2_roundabout):
            return
        if self.route_override == "none":
            return
        if self.route_override in {"left", "right"}:
            detected_intent = self.route_override
            self.route_override = "auto"
            if self.first_route is None:
                self._begin_roundabout(detected_intent, 1, now_sec)
            else:
                self._begin_roundabout(detected_intent, 2, now_sec)
            return
        elif self.route_override == "clear":
            self.route_override = "auto"
            self.clear_route()
            return
        elif not self._map_allows("route_sign"):
            return
        elif self._confirmed("blue_left_turn_sign", now_sec):
            detected_intent = "left"
        elif self._confirmed("blue_right_turn_sign", now_sec):
            detected_intent = "right"

        if detected_intent is None:
            return
        route_index = 2 if self.mission_phase == "RA2_ROUTE_SIGN_ZONE" else 1
        if route_index == 1 and not self.enable_ra1_route_sign:
            return
        if route_index == 2 and not self.enable_ra2_route_sign:
            return
        if route_index == 1 and now_sec - self.last_roundabout_sign_time < self.roundabout_sign_cooldown_sec:
            return
        self._consume("blue_left_turn_sign", now_sec, self.roundabout_sign_cooldown_sec)
        self._consume("blue_right_turn_sign", now_sec, self.roundabout_sign_cooldown_sec)
        self._begin_roundabout(detected_intent, route_index, now_sec)
        self.last_roundabout_sign_time = now_sec

    def _begin_roundabout(self, route: str, index: int, now_sec: float) -> None:
        if route not in {"left", "right"}:
            return
        if self.state in ROUNDABOUT_STATES:
            return
        if index == 1:
            self.first_route = route
            self.second_route = "right" if route == "left" else "left"
        elif index == 2:
            self.second_route = route
        if not self._roundabout_task_enabled(index):
            if index == 1:
                self._set_phase("SIDE_SIGN_ZONE_1", now_sec)
            else:
                self._set_phase("FINAL_STRAIGHT", now_sec)
            return
        self.route_intent = route
        self.roundabout_exit_index = 3 if route == "left" else 1
        self.active_roundabout_index = index
        self.state = "ROUNDABOUT_APPROACH"
        self._set_phase(f"RA{index}_ENTRY", now_sec)
        self.roundabout_phase_started_at = now_sec

    def _vehicle_clear(self, now_sec: float) -> bool:
        if self.vehicle_clear_since is None:
            self.vehicle_clear_since = now_sec
            return False
        return now_sec - self.vehicle_clear_since >= self.roundabout_yield_clear_sec

    def _traffic_planning_window_active(self, now_sec: float) -> bool:
        if self.traffic_first_seen_at is None:
            return False
        window_sec = max(0.0, self.traffic_planning_window_sec)
        return window_sec > 0.0 and now_sec - self.traffic_first_seen_at <= window_sec

    def _remember_traffic_first_seen(self, now_sec: float) -> None:
        if self.traffic_first_seen_at is None or not self._traffic_planning_window_active(now_sec):
            self.traffic_first_seen_at = now_sec
            self.traffic_clear_since = None
            self.traffic_red_latched = False
            self.traffic_passed = False

    def _finish_traffic_phase(self, now_sec: float) -> None:
        self.traffic_clear_since = None
        self.traffic_red_latched = False
        self.traffic_passed = True
        self._set_phase("POST_TRAFFIC_LANE_FOLLOW", now_sec, reset_timer=False)

    def _set_phase(self, phase: str, now_sec: float, reset_timer: bool = True) -> None:
        if phase not in MISSION_PHASES:
            return
        if self.mission_phase != phase:
            self.mission_phase = phase
            self.mission_phase_started_at = now_sec
        elif reset_timer:
            self.mission_phase_started_at = now_sec

    def _advance_timed_mission(self, now_sec: float) -> None:
        elapsed = now_sec - self.mission_phase_started_at
        if self.state in ROUNDABOUT_STATES:
            return
        if self.mission_phase == "RA1_EXITED" and elapsed >= self.roundabout_to_side_sec:
            self._set_phase("SIDE_SIGN_ZONE_1", now_sec)
        elif self.mission_phase == "SIDE_SIGN_ZONE_1" and elapsed >= self.side_sign_zone_timeout_sec:
            self._set_phase("POST_SIGN_LANE_FOLLOW", now_sec)
        elif self.mission_phase == "POST_SIGN_LANE_FOLLOW" and elapsed >= self.post_sign_lane_follow_sec:
            self._set_phase("TRAFFIC_LIGHT_ZONE", now_sec)
        elif self.mission_phase == "POST_TRAFFIC_LANE_FOLLOW" and elapsed >= self.post_traffic_lane_follow_sec:
            if self.enable_ra2_route_sign:
                self._set_phase("RA2_ROUTE_SIGN_ZONE", now_sec)
            else:
                self._start_second_roundabout(now_sec)
        elif self.mission_phase == "INTERSECTION_CROSS" and elapsed >= self.intersection_cross_sec:
            self._set_phase("SIDE_SIGN_ZONE_2", now_sec)
        elif self.mission_phase == "SIDE_SIGN_ZONE_2" and elapsed >= self.side_sign_zone_timeout_sec:
            self._start_second_roundabout(now_sec)
        elif self.mission_phase == "RA2_EXITED" and elapsed >= self.roundabout_to_side_sec:
            self._set_phase("FINAL_STRAIGHT", now_sec)

    def _finish_side_sign_zone(self, now_sec: float) -> None:
        if self.mission_phase in {"RA1_EXITED", "SIDE_SIGN_ZONE_1"}:
            self._set_phase("POST_SIGN_LANE_FOLLOW", now_sec)
        elif self.mission_phase == "SIDE_SIGN_ZONE_2":
            self._start_second_roundabout(now_sec)

    def _start_second_roundabout(self, now_sec: float) -> None:
        if self.state in ROUNDABOUT_STATES:
            return
        if not self.enable_ra2_roundabout:
            self._set_phase("FINAL_STRAIGHT", now_sec)
            return
        route = self.second_route
        if route is None and self.first_route is not None:
            route = "right" if self.first_route == "left" else "left"
            self.second_route = route
        if route is not None:
            self._begin_roundabout(route, 2, now_sec)
        else:
            self._set_phase("FINAL_STRAIGHT", now_sec)

    def _biased_roundabout_phase_steer(self, steer: float) -> float:
        hint = self.lane_route_hint()
        if hint == "approach":
            bias = self.roundabout_approach_steer_bias
        elif hint == "left_circulate":
            bias = self.roundabout_left_circulate_steer_bias
        elif hint == "right_circulate":
            bias = self.roundabout_right_circulate_steer_bias
        elif hint == "left_exit":
            bias = self.roundabout_left_exit_steer_bias
        elif hint == "right_exit":
            bias = self.roundabout_right_exit_steer_bias
        else:
            bias = 0.0
        bias = max(-self.max_steer_bias, min(self.max_steer_bias, bias))
        return max(-1.0, min(1.0, steer + bias))

    def _roundabout_exit_duration(self) -> float:
        if self.active_roundabout_index == 2 and self.route_intent == "left":
            return max(0.0, self.roundabout_ra2_left_exit_sec)
        if self.active_roundabout_index == 2 and self.route_intent == "right":
            return max(0.0, self.roundabout_ra2_right_exit_sec)
        return max(0.0, self.roundabout_exit_sec)

    def _biased_circulate_steer(self, steer: float) -> float:
        if self.active_roundabout_index == 1:
            bias = self.roundabout_ra1_circulate_bias
        elif self.active_roundabout_index == 2:
            bias = self.roundabout_ra2_circulate_bias
        else:
            bias = 0.0
        bias = max(-self.max_steer_bias, min(self.max_steer_bias, bias))
        return max(-1.0, min(1.0, steer + bias))

    def _roundabout_lane_lost_recovery_enabled(self) -> bool:
        return (
            self.state in {"ROUNDABOUT_IN", "ROUNDABOUT_EXIT"}
            and self.active_roundabout_index > 0
            and self._roundabout_task_enabled(self.active_roundabout_index)
            and abs(self.roundabout_lane_lost_recovery_steer) > 1e-6
            and self.roundabout_lane_lost_recovery_speed > 0.0
        )

    def _roundabout_lane_lost_recovery_command(
        self,
        lane_state: str,
        stable: set,
        lane_valid: bool,
        lane_confidence: float,
        now_sec: float,
    ) -> Dict[str, object]:
        self._set_phase(f"RA{self.active_roundabout_index}_INSIDE", now_sec, reset_timer=False)
        phase = "exit" if self.state == "ROUNDABOUT_EXIT" else "inside"
        reason = f"roundabout_lane_lost_left_recovery_{phase}:{lane_state}"
        return self._command(
            steer=self.roundabout_lane_lost_recovery_steer,
            speed=min(max(0.0, self.roundabout_lane_lost_recovery_speed), max(0.0, self.roundabout_speed_limit)),
            brake=False,
            reason=reason,
            stable=stable,
            lane_valid=lane_valid,
            lane_confidence=lane_confidence,
            allow_lane_fault_command=True,
        )

    def _lane_lost_scan_recovery_enabled(self, stable: set) -> bool:
        if self.state in ROUNDABOUT_STATES:
            return False
        if self.state in {"STOP_SIGN_HOLD", "TRAFFIC_LIGHT_WAIT", "ROUNDABOUT_ENTRY_YIELD"}:
            return False
        if self.traffic_red_latched and not self.traffic_passed:
            return False
        if "traffic_light_red" in stable:
            return False
        return (
            self.lane_lost_scan_turn_speed > 0.0
            and (self.lane_lost_scan_right_sec > 0.0 or self.lane_lost_scan_left_sec > 0.0)
        )

    def _lane_lost_scan_recovery_command(
        self,
        lane_state: str,
        stable: set,
        lane_valid: bool,
        lane_confidence: float,
        now_sec: float,
    ) -> Dict[str, object]:
        if self.lane_lost_scan_direction == 0 or self.lane_lost_recovery_started_at is None:
            self._choose_random_lane_lost_scan_direction(now_sec)

        phase_target = self._lane_lost_scan_target_sec(self.lane_lost_scan_direction)
        segment_started_at = self.lane_lost_recovery_started_at
        phase_time = now_sec - segment_started_at if segment_started_at is not None else 0.0
        if phase_time >= phase_target:
            self._choose_random_lane_lost_scan_direction(now_sec)
            phase_target = self._lane_lost_scan_target_sec(self.lane_lost_scan_direction)
            phase_time = 0.0

        steer = 1.0 if self.lane_lost_scan_direction >= 0 else -1.0
        phase = "right" if steer > 0.0 else "left"
        speed = max(0.0, self.lane_lost_scan_turn_speed)
        return self._command(
            steer=steer,
            speed=speed,
            brake=False,
            reason=f"lane_lost_scan_{phase}:{lane_state}:{phase_time:.2f}/{phase_target:.2f}",
            stable=stable,
            lane_valid=lane_valid,
            lane_confidence=lane_confidence,
            allow_lane_fault_command=True,
            force_in_place_turn=True,
        )

    def _choose_random_lane_lost_scan_direction(self, now_sec: float) -> None:
        left_weight, right_weight = self._lane_lost_scan_direction_weights()
        weighted_directions = []
        weights = []
        if self.lane_lost_scan_left_sec > 0.0:
            weighted_directions.append(-1)
            weights.append(left_weight)
        if self.lane_lost_scan_right_sec > 0.0:
            weighted_directions.append(1)
            weights.append(right_weight)
        if weighted_directions and any(weight > 0.0 for weight in weights):
            self.lane_lost_scan_direction = random.choices(weighted_directions, weights=weights, k=1)[0]
        else:
            directions = []
            if self.lane_lost_scan_right_sec > 0.0:
                directions.append(1)
            if self.lane_lost_scan_left_sec > 0.0:
                directions.append(-1)
            self.lane_lost_scan_direction = random.choice(directions) if directions else 0
        self.lane_lost_recovery_started_at = now_sec

    def _lane_lost_scan_direction_weights(self) -> Tuple[float, float]:
        if self._second_roundabout_started():
            return 1.0, 1.0
        if self.first_route == "right":
            return (
                max(0.0, self.lane_lost_scan_first_right_left_weight),
                max(0.0, self.lane_lost_scan_first_right_right_weight),
            )
        if self.first_route == "left":
            return (
                max(0.0, self.lane_lost_scan_first_left_left_weight),
                max(0.0, self.lane_lost_scan_first_left_right_weight),
            )
        return 1.0, 1.0

    def _second_roundabout_started(self) -> bool:
        return self.active_roundabout_index == 2 or self.mission_phase in {
            "RA2_ENTRY",
            "RA2_INSIDE",
            "RA2_EXITED",
            "FINAL_STRAIGHT",
        }

    def _lane_lost_scan_target_sec(self, direction: int) -> float:
        if direction < 0:
            return max(0.1, self.lane_lost_scan_left_sec)
        return max(0.1, self.lane_lost_scan_right_sec)

    def _reset_lane_lost_recovery(self) -> None:
        self.lane_lost_recovery_started_at = None
        self.lane_lost_scan_direction = 0

    def _update_detection_memory(
        self,
        detections: Iterable[Detection],
        now_sec: float,
        frame_shape: Optional[Tuple[int, int]],
        detections_fresh: bool,
    ) -> None:
        if not detections_fresh:
            return
        best_by_class: Dict[str, DetectionSnapshot] = {}
        for det in detections:
            snapshot = self._snapshot_if_relevant(det, frame_shape)
            if snapshot is None:
                continue
            current = best_by_class.get(snapshot.class_name)
            # 같은 클래스는 confidence*area가 큰 박스를 사용
            if current is None or snapshot.confidence * max(snapshot.adjusted_area_ratio, 0.001) > current.confidence * max(current.adjusted_area_ratio, 0.001):
                best_by_class[snapshot.class_name] = snapshot

        tracked_names = set(CLASS_NAMES.values())
        for class_name in tracked_names:
            memory = self.detection_memory.setdefault(class_name, DetectionMemory())
            memory.update(
                best_by_class.get(class_name),
                now_sec,
                max_misses=max(0, int(self.detection_max_misses)),
                max_hits=max(3, int(self.detection_confirm_hits) + 3),
            )

    def _snapshot_if_relevant(
        self,
        det: Detection,
        frame_shape: Optional[Tuple[int, int]],
    ) -> Optional[DetectionSnapshot]:
        frame_w = 1280.0
        frame_h = 720.0
        if frame_shape is not None and len(frame_shape) >= 2:
            frame_h = max(1.0, float(frame_shape[0]))
            frame_w = max(1.0, float(frame_shape[1]))
        class_name = det.class_name
        x1, y1, x2, y2 = det.xyxy
        box_w = max(1e-6, x2 - x1)
        box_h = max(1e-6, y2 - y1)
        aspect_ratio = max(box_w / box_h, box_h / box_w)
        cx_ratio = ((x1 + x2) * 0.5) / frame_w
        bottom_ratio = y2 / frame_h
        height_ratio = box_h / frame_h
        area_ratio = max(0.0, box_w * box_h) / max(1.0, frame_w * frame_h)
        distance_scale = max(0.10, self.camera_distance_scale)
        # 카메라 위치/FOV 보정값
        adjusted_area_ratio = area_ratio / (distance_scale * distance_scale)
        adjusted_height_ratio = height_ratio / distance_scale
        adjusted_bottom_ratio = 1.0 - ((1.0 - bottom_ratio) * distance_scale)
        adjusted_bottom_ratio = max(0.0, min(1.0, adjusted_bottom_ratio))

        if class_name in {"blue_left_turn_sign", "blue_right_turn_sign"}:
            if not (self.enable_ra1_route_sign or self.enable_ra2_route_sign):
                return None
            if det.confidence < self.sign_conf:
                return None
            if adjusted_area_ratio < self.route_sign_min_area or adjusted_bottom_ratio < self.route_sign_min_bottom_y:
                return None
        elif class_name == "red_stop_sign":
            if not self.enable_stop_sign:
                return None
            if det.confidence < self.sign_conf:
                return None
            if adjusted_area_ratio < self.stop_sign_min_area or adjusted_bottom_ratio < self.stop_sign_min_bottom_y:
                return None
        elif class_name == "yellow_pedestrian_warning_sign":
            if not self.enable_pedestrian_sign:
                return None
            if det.confidence < self.sign_conf:
                return None
            if adjusted_area_ratio < self.pedestrian_sign_min_area or adjusted_bottom_ratio < self.pedestrian_sign_min_bottom_y:
                return None
        elif class_name in {"traffic_light_red", "traffic_light_orange", "traffic_light_green", "traffic_light"}:
            if not self.enable_traffic_light:
                return None
            min_conf = self.traffic_green_conf if class_name == "traffic_light_green" else self.traffic_stop_conf
            if det.confidence < min_conf:
                return None
            if class_name != "traffic_light":
                max_aspect = max(0.0, self.traffic_color_max_aspect_ratio)
                if max_aspect > 0.0 and aspect_ratio > max_aspect:
                    return None
            min_area = self.traffic_color_min_area if class_name != "traffic_light" else self.traffic_light_min_area
            if adjusted_area_ratio < min_area or adjusted_bottom_ratio < self.traffic_light_min_bottom_y:
                return None
        elif class_name == "toy_car":
            if not self.enable_toy_car_yield:
                return None
            if det.confidence < self.obstacle_stop_conf:
                return None
            centered = self.obstacle_x_min <= cx_ratio <= self.obstacle_x_max
            height_min = max(0.0, self.obstacle_height_min)
            tall_enough = height_min <= 0.0 or adjusted_height_ratio >= height_min
            # 중앙에 있고 크거나 낮게 보이면 장애물로 판단
            close_enough = (
                adjusted_area_ratio >= self.obstacle_area_min
                or adjusted_bottom_ratio >= self.obstacle_bottom_y_min
            )
            if not centered or not tall_enough or not close_enough:
                return None
        else:
            return None

        return DetectionSnapshot(
            class_name=class_name,
            confidence=det.confidence,
            xyxy=list(det.xyxy),
            area_ratio=area_ratio,
            bottom_y_ratio=bottom_ratio,
            center_x_ratio=cx_ratio,
            adjusted_area_ratio=adjusted_area_ratio,
            adjusted_bottom_y_ratio=adjusted_bottom_ratio,
        )

    def _confirmed(self, class_name: str, now_sec: float) -> bool:
        memory = self.detection_memory.get(class_name)
        if memory is None:
            return False
        return memory.confirmed(
            now_sec,
            min_hits=max(1, int(self.detection_confirm_hits)),
            hold_sec=max(0.0, self.detection_memory_hold_sec),
        )

    def _last_seen(self, class_name: str) -> float:
        memory = self.detection_memory.get(class_name)
        if memory is None:
            return -999.0
        return memory.last_seen_sec

    def _confirmed_classes(self, now_sec: float) -> set:
        return {class_name for class_name in self.detection_memory if self._confirmed(class_name, now_sec)}

    def _consume(self, class_name: str, now_sec: float, cooldown_sec: float) -> None:
        memory = self.detection_memory.get(class_name)
        if memory is not None:
            memory.consume(now_sec, cooldown_sec)

    def _confirmed_vehicle_hazard(self, now_sec: float) -> bool:
        return self.enable_toy_car_yield and self._confirmed("toy_car", now_sec)

    def _roundabout_task_enabled(self, index: int) -> bool:
        if index == 1:
            return self.enable_ra1_roundabout
        if index == 2:
            return self.enable_ra2_roundabout
        return self.enable_ra1_roundabout or self.enable_ra2_roundabout

    def _skip_disabled_mission_tasks(self, now_sec: float) -> None:
        for _ in range(8):
            phase = self.mission_phase
            if phase == "RA1_ROUTE_SIGN_ZONE" and (not self.enable_ra1_route_sign or not self.enable_ra1_roundabout):
                self._set_phase("SIDE_SIGN_ZONE_1", now_sec)
                continue
            if phase in {"RA1_EXITED", "SIDE_SIGN_ZONE_1"} and not self.enable_side_sign_1:
                self._set_phase("POST_SIGN_LANE_FOLLOW", now_sec)
                continue
            if phase == "POST_SIGN_LANE_FOLLOW" and not self.enable_traffic_light:
                self._set_phase("POST_TRAFFIC_LANE_FOLLOW", now_sec)
                continue
            if phase in {"TRAFFIC_LIGHT_ZONE", "INTERSECTION_CROSS"} and not self.enable_traffic_light:
                self._set_phase("POST_TRAFFIC_LANE_FOLLOW", now_sec)
                continue
            if phase == "POST_TRAFFIC_LANE_FOLLOW" and not self.enable_ra2_roundabout:
                self._set_phase("FINAL_STRAIGHT", now_sec)
                continue
            if phase == "RA2_ROUTE_SIGN_ZONE" and (not self.enable_ra2_route_sign or not self.enable_ra2_roundabout):
                if self.enable_ra2_roundabout:
                    self._start_second_roundabout(now_sec)
                else:
                    self._set_phase("FINAL_STRAIGHT", now_sec)
                continue
            if phase == "SIDE_SIGN_ZONE_2" and not self.enable_side_sign_2:
                if self.enable_ra2_roundabout:
                    self._start_second_roundabout(now_sec)
                else:
                    self._set_phase("FINAL_STRAIGHT", now_sec)
                continue
            if phase in {"RA2_ENTRY", "RA2_INSIDE", "RA2_EXITED"} and not self.enable_ra2_roundabout:
                self._set_phase("FINAL_STRAIGHT", now_sec)
                continue
            break

    def _map_allows(self, event_name: str) -> bool:
        if event_name == "route_sign":
            if self.mission_phase == "RA1_ROUTE_SIGN_ZONE" and not self.enable_ra1_route_sign:
                return False
            if self.mission_phase == "RA2_ROUTE_SIGN_ZONE" and not self.enable_ra2_route_sign:
                return False
            if self.mission_phase not in {"RA1_ROUTE_SIGN_ZONE", "RA2_ROUTE_SIGN_ZONE"}:
                return False
        if event_name == "side_sign":
            if self.mission_phase in {"RA1_EXITED", "SIDE_SIGN_ZONE_1"} and not self.enable_side_sign_1:
                return False
            if self.mission_phase == "SIDE_SIGN_ZONE_2" and not self.enable_side_sign_2:
                return False
        if event_name == "traffic_light" and not self.enable_traffic_light:
            return False
        if self.map_gate_mode == "off":
            return True
        # 현재 미션 구간에서 허용되는 이벤트만 통과
        if event_name == "route_sign":
            allowed = self.mission_phase in {"RA1_ROUTE_SIGN_ZONE", "RA2_ROUTE_SIGN_ZONE"}
        elif event_name == "side_sign":
            allowed = self.mission_phase in {"RA1_EXITED", "SIDE_SIGN_ZONE_1", "SIDE_SIGN_ZONE_2"}
        elif event_name == "traffic_light":
            allowed = self.mission_phase in {"TRAFFIC_LIGHT_ZONE", "INTERSECTION_CROSS"}
        else:
            allowed = True
        if allowed:
            return True
        return self.map_gate_mode == "soft" and event_name == "traffic_light"


def detections_from_ultralytics(result) -> List[Detection]:
    detections: List[Detection] = []
    for box in result.boxes:
        detections.append(
            Detection(
                class_id=int(box.cls[0].item()),
                confidence=float(box.conf[0].item()),
                xyxy=[float(v) for v in box.xyxy[0].tolist()],
            )
        )
    return detections
