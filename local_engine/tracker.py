"""State-of-the-Art Dual Tracker: MediaPipe Hands (21 pts) + MediaPipe Pose (33 pts)."""
import os
import time
import urllib.request
import math
import cv2
import numpy as np
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_PATH = os.path.join(_DIR, "hand_landmarker.task")
POSE_PATH = os.path.join(_DIR, "pose_landmarker_lite.task")

def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

class OneEuroFilter:
    def __init__(self, x0: float = 0.0, dx0: float = 0.0, min_cutoff: float = 0.5, beta: float = 2.5, d_cutoff: float = 1.0):
        self.x = x0
        self.dx = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.t_prev = time.time()

    def __call__(self, x: float) -> float:
        t = time.time()
        t_e = max(t - self.t_prev, 1e-6)
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x
        self.x = x_hat
        self.dx = dx_hat
        self.t_prev = t
        return x_hat

@dataclass
class HandResult:
    palm: Tuple[float, float]
    index_mcp: Tuple[float, float]
    pinky_mcp: Tuple[float, float]
    wrist: Tuple[float, float]
    middle_mcp: Tuple[float, float]
    hand_angle: float
    handedness: str
    confidence: float

@dataclass
class PoseResult:
    nose: Optional[Tuple[float, float]] = None
    left_eye: Optional[Tuple[float, float]] = None
    right_eye: Optional[Tuple[float, float]] = None
    left_ear: Optional[Tuple[float, float]] = None
    right_ear: Optional[Tuple[float, float]] = None
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    left_elbow: Optional[Tuple[float, float]] = None
    right_elbow: Optional[Tuple[float, float]] = None
    left_wrist_pose: Optional[Tuple[float, float]] = None
    right_wrist_pose: Optional[Tuple[float, float]] = None
    left_hip: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    head_center: Optional[Tuple[float, float]] = None
    head_angle: Optional[float] = None
    ear_distance: Optional[float] = None
    neck_point: Optional[Tuple[float, float]] = None
    shoulder_width: Optional[float] = None
    shoulder_angle: Optional[float] = None
    left_forearm_length: Optional[float] = None
    right_forearm_length: Optional[float] = None
    left_ear_eye_dist: Optional[float] = None
    right_ear_eye_dist: Optional[float] = None
    torso_height: Optional[float] = None

@dataclass
class TrackResult:
    hands: List[HandResult]
    pose: Optional[PoseResult]

class _HandFilterBank:
    def __init__(self):
        mc, b = 0.5, 2.5
        self.palm_x  = OneEuroFilter(0.5, min_cutoff=mc, beta=b)
        self.palm_y  = OneEuroFilter(0.5, min_cutoff=mc, beta=b)
        self.idx_x   = OneEuroFilter(0.5, min_cutoff=mc, beta=b)
        self.idx_y   = OneEuroFilter(0.5, min_cutoff=mc, beta=b)
        self.pinky_x = OneEuroFilter(0.5, min_cutoff=mc, beta=b)
        self.pinky_y = OneEuroFilter(0.5, min_cutoff=mc, beta=b)
        self.angle_sin = OneEuroFilter(0.0, min_cutoff=0.2, beta=0.3)
        self.angle_cos = OneEuroFilter(1.0, min_cutoff=0.2, beta=0.3)
        self.last_update = time.time()

class _PoseFilterBank:
    def __init__(self):
        mc, b = 0.5, 2.5
        def _p(x0: float = 0.5) -> OneEuroFilter: return OneEuroFilter(x0, min_cutoff=mc, beta=b)
        self.nose_x, self.nose_y   = _p(), _p()
        self.leye_x, self.leye_y   = _p(), _p()
        self.reye_x, self.reye_y   = _p(), _p()
        self.lear_x, self.lear_y   = _p(), _p()
        self.rear_x, self.rear_y   = _p(), _p()
        self.lsho_x, self.lsho_y   = _p(), _p()
        self.rsho_x, self.rsho_y   = _p(), _p()
        self.lelb_x, self.lelb_y   = _p(), _p()
        self.relb_x, self.relb_y   = _p(), _p()
        self.lwri_x, self.lwri_y   = _p(), _p()
        self.rwri_x, self.rwri_y   = _p(), _p()
        self.lhip_x, self.lhip_y   = _p(), _p()
        self.rhip_x, self.rhip_y   = _p(), _p()
        self.head_angle     = OneEuroFilter(0.0, min_cutoff=0.3, beta=0.5)
        self.shoulder_angle = OneEuroFilter(0.0, min_cutoff=0.3, beta=0.5)

class UltimateTracker:
    _VIS_THRESH = 0.3

    def __init__(self, *, num_hands: int = 2, min_detection_confidence: float = 0.6, min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.55):
        if not os.path.exists(HAND_PATH): urllib.request.urlretrieve(HAND_URL, HAND_PATH)
        if not os.path.exists(POSE_PATH): urllib.request.urlretrieve(POSE_URL, POSE_PATH)

        hand_opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=HAND_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence)
        pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=POSE_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)

        self.hand_tracker = mp.tasks.vision.HandLandmarker.create_from_options(hand_opts)
        self.pose_tracker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts)
        self._hand_filters: Dict[str, _HandFilterBank] = {"Left": _HandFilterBank(), "Right": _HandFilterBank()}
        self._pose_fb = _PoseFilterBank()
        self._last_ts: Optional[int] = None
        # Hand + pose are independent models — run them concurrently instead
        # of one-after-another. Both release the GIL during C++ inference,
        # so this actually overlaps their compute instead of just adding
        # thread overhead.
        self._pool = ThreadPoolExecutor(max_workers=2)

    def detect(self, frame_bgr: np.ndarray) -> TrackResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        t = int(time.monotonic() * 1000)
        if self._last_ts is None: self._last_ts = t
        elif t <= self._last_ts: t = self._last_ts + 1
        self._last_ts = t

        hand_future = self._pool.submit(self.hand_tracker.detect_for_video, mp_image, t)
        pose_future = self._pool.submit(self.pose_tracker.detect_for_video, mp_image, t)
        hands = self._process_hands(hand_future.result())
        pose = self._process_pose(pose_future.result())
        return TrackResult(hands=hands, pose=pose)

    def _process_hands(self, hand_result) -> List[HandResult]:
        hands: List[HandResult] = []
        if not hand_result.hand_landmarks: return hands
        for i, hand_lm in enumerate(hand_result.hand_landmarks):
            handedness_label = "Right"
            confidence = 0.5
            if hand_result.handedness and i < len(hand_result.handedness):
                info = hand_result.handedness[i][0]
                handedness_label = info.category_name
                confidence = info.score
            if confidence < 0.4: continue
            fb = self._hand_filters.get(handedness_label, _HandFilterBank())
            self._hand_filters[handedness_label] = fb

            wrist, index_mcp, middle_mcp, pinky_mcp = hand_lm[0], hand_lm[5], hand_lm[9], hand_lm[17]
            palm_raw_x = 0.4 * wrist.x + 0.3 * index_mcp.x + 0.3 * pinky_mcp.x
            palm_raw_y = 0.4 * wrist.y + 0.3 * index_mcp.y + 0.3 * pinky_mcp.y
            dx, dy = middle_mcp.x - wrist.x, middle_mcp.y - wrist.y
            raw_angle = math.atan2(dy, dx)
            s = fb.angle_sin(math.sin(raw_angle))
            c = fb.angle_cos(math.cos(raw_angle))
            smoothed_angle = math.atan2(s, c)
            fb.last_update = time.time()

            hands.append(HandResult(
                palm=(fb.palm_x(palm_raw_x), fb.palm_y(palm_raw_y)),
                index_mcp=(fb.idx_x(index_mcp.x), fb.idx_y(index_mcp.y)),
                pinky_mcp=(fb.pinky_x(pinky_mcp.x), fb.pinky_y(pinky_mcp.y)),
                wrist=(wrist.x, wrist.y), middle_mcp=(middle_mcp.x, middle_mcp.y),
                hand_angle=smoothed_angle, handedness=handedness_label, confidence=confidence))
        return hands

    def _process_pose(self, pose_result) -> Optional[PoseResult]:
        if not pose_result.pose_landmarks: return None
        lm = pose_result.pose_landmarks[0]
        fb = self._pose_fb
        VT = self._VIS_THRESH
        def _get(idx, fx, fy): 
            pt = lm[idx]
            return (fx(pt.x), fy(pt.y)) if pt.visibility > VT else None

        r = PoseResult()
        r.nose, r.left_eye, r.right_eye = _get(0, fb.nose_x, fb.nose_y), _get(2, fb.leye_x, fb.leye_y), _get(5, fb.reye_x, fb.reye_y)
        r.left_ear, r.right_ear = _get(7, fb.lear_x, fb.lear_y), _get(8, fb.rear_x, fb.rear_y)
        r.left_shoulder, r.right_shoulder = _get(11, fb.lsho_x, fb.lsho_y), _get(12, fb.rsho_x, fb.rsho_y)
        r.left_elbow, r.right_elbow = _get(13, fb.lelb_x, fb.lelb_y), _get(14, fb.relb_x, fb.relb_y)
        r.left_wrist_pose, r.right_wrist_pose = _get(15, fb.lwri_x, fb.lwri_y), _get(16, fb.rwri_x, fb.rwri_y)
        r.left_hip, r.right_hip = _get(23, fb.lhip_x, fb.lhip_y), _get(24, fb.rhip_x, fb.rhip_y)

        if r.left_ear and r.right_ear:
            r.head_center = ((r.left_ear[0] + r.right_ear[0]) / 2, (r.left_ear[1] + r.right_ear[1]) / 2)
            r.ear_distance = math.dist(r.left_ear, r.right_ear)
            r.head_angle = fb.head_angle(math.degrees(math.atan2(r.right_ear[1] - r.left_ear[1], r.right_ear[0] - r.left_ear[0])))
        if r.left_shoulder and r.right_shoulder:
            sho_mid = ((r.left_shoulder[0] + r.right_shoulder[0]) / 2, (r.left_shoulder[1] + r.right_shoulder[1]) / 2)
            r.shoulder_width = math.dist(r.left_shoulder, r.right_shoulder)
            r.shoulder_angle = fb.shoulder_angle(math.degrees(math.atan2(r.right_shoulder[1] - r.left_shoulder[1], r.right_shoulder[0] - r.left_shoulder[0])))
            r.neck_point = (sho_mid[0] + 0.3 * (r.head_center[0] - sho_mid[0]), sho_mid[1] + 0.3 * (r.head_center[1] - sho_mid[1])) if r.head_center else sho_mid
        if r.left_elbow and r.left_wrist_pose: r.left_forearm_length = math.dist(r.left_elbow, r.left_wrist_pose)
        if r.right_elbow and r.right_wrist_pose: r.right_forearm_length = math.dist(r.right_elbow, r.right_wrist_pose)
        if r.left_shoulder and r.right_shoulder and r.left_hip and r.right_hip:
            sho_mid = ((r.left_shoulder[0] + r.right_shoulder[0]) / 2, (r.left_shoulder[1] + r.right_shoulder[1]) / 2)
            hip_mid = ((r.left_hip[0] + r.right_hip[0]) / 2, (r.left_hip[1] + r.right_hip[1]) / 2)
            r.torso_height = math.dist(sho_mid, hip_mid)
        elif r.left_shoulder and r.right_shoulder and r.shoulder_width: r.torso_height = r.shoulder_width * 2.5
        if r.left_ear and r.left_eye: r.left_ear_eye_dist = math.dist(r.left_ear, r.left_eye)
        if r.right_ear and r.right_eye: r.right_ear_eye_dist = math.dist(r.right_ear, r.right_eye)
        return r