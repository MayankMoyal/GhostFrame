"""Shared pipeline: webcam frame → pose landmarks → resolved anchor →
spatial transform → smoothed overlay payload.

Both ``pose_test.py`` (local debug window) and ``server.py``
(FastAPI/WebSocket service) import this so the actual pose/anchor/
smoothing logic exists in exactly one place.

One ``PoseAnchorPipeline`` instance per consumer (one per debug window
run, one per connected websocket client on the server) — state like
the smoother, frame timestamp counter, and last-resolved-type tracker
must not be shared across independent streams.
"""
import os
from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .anchor_resolver import resolve_anchor, compute_spatial_transform
from .pose_filter import SpatialSmoother

_OVERLAY_DIR = Path(__file__).resolve().parent
_MODEL_CACHE = _OVERLAY_DIR / "models"
MODEL_PATH = _MODEL_CACHE / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Tracked keypoints — includes hips for skeleton drawing and
# potential future fallback logic.
ANCHOR_KEYPOINTS = {
    "nose": 0,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
}

# Full skeleton connection list for debug rendering.
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),        # Left face
    (0, 4), (4, 5), (5, 6), (6, 8),        # Right face
    (9, 10),                                # Mouth
    (11, 12), (11, 13), (13, 15),           # Left arm
    (12, 14), (14, 16),                     # Right arm
    (11, 23), (12, 24), (23, 24),           # Torso
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # Right leg
]


def ensure_model():
    """Download the MediaPipe Pose Landmarker model if not cached."""
    if not os.path.exists(MODEL_PATH):
        _MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        print("Downloading MediaPipe model...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))


def extract_lighting(frame, points_px=None):
    """Extract luminance and tint from the webcam frame.

    If anchor pixel coordinates are provided, samples a 96×96 patch
    around the anchor instead of the full frame — this gives more
    accurate ambient-matching when the streamer is lit differently
    from the background.

    Parameters
    ----------
    frame : np.ndarray
        BGR webcam frame.
    points_px : list[dict] | None
        Resolved anchor point(s) in pixel coordinates.

    Returns
    -------
    dict
        ``brightness`` (CSS multiplier 0.6-1.4) and ``tint`` (hex).
    """
    h, w, _ = frame.shape
    patch = frame

    if points_px and len(points_px) > 0:
        cx = sum(p["x"] for p in points_px) / len(points_px)
        cy = sum(p["y"] for p in points_px) / len(points_px)
        px, py = int(cx), int(cy)
        patch_size = 96

        x1 = max(0, px - patch_size // 2)
        y1 = max(0, py - patch_size // 2)
        x2 = min(w, px + patch_size // 2)
        y2 = min(h, py + patch_size // 2)

        local_patch = frame[y1:y2, x1:x2]
        if local_patch.size > 0:
            patch = local_patch

    small = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_AREA)
    mean_bgr = np.mean(small, axis=(0, 1))
    b, g, r = mean_bgr[0], mean_bgr[1], mean_bgr[2]

    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    brightness_css = round(max(0.6, min(1.4, luminance * 1.5)), 2)
    tint_hex = f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    return {"brightness": brightness_css, "tint": tint_hex}


def extract_keypoints(landmarks):
    """Extract named keypoints from MediaPipe landmarks.

    ``z`` is MediaPipe's hip-relative depth estimate (negative =
    closer to camera).  ``visibility`` is the model's confidence
    that the joint is present in the frame.
    """
    keypoints = {}
    for name, idx in ANCHOR_KEYPOINTS.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            keypoints[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
    return keypoints


def build_overlay_payload(smooth_tf, lighting, resolved, fps):
    """Final overlay data dict, JSON-serializable as-is.

    Payload schema
    --------------
    .. code-block:: json

        {
            "points":      [{"x": float, "y": float}, ...],
            "angle":       float,
            "scale":       float,
            "brightness":  float,
            "tint":        str,
            "anchor_type": str,
            "fallback":    bool,
            "fps":         float,
            "timestamp":   float
        }

    ``points`` is a list of 1 entry for single/ambient anchors, or
    2 entries for dual anchors (both raw coordinates, no midpoint
    collapse).
    """
    return {
        "points":      smooth_tf["points"],
        "angle":       smooth_tf["angle"],
        "scale":       smooth_tf["scale"],
        "brightness":  smooth_tf.get("brightness", lighting["brightness"]),
        "tint":        lighting["tint"],
        "anchor_type": resolved["resolved_type"],
        "fallback":    resolved["fallback_applied"],
        "fps":         round(fps, 1),
        "timestamp":   time.time(),
    }


class PoseAnchorPipeline:
    """Stateful per-connection pipeline: feed it frames, get overlay payloads.

    One instance per consumer (one per debug window run, one per
    connected websocket client on the server) — state like the
    smoother, frame timestamp counter, and last-resolved-type tracker
    must not be shared across independent streams.
    """

    def __init__(self, anchor_type="background"):
        ensure_model()
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.smoother = SpatialSmoother()
        self.anchor_type = anchor_type
        self.frame_timestamp_ms = 0
        self.prev_time = time.time()
        self.last_resolved_type = None

    def set_anchor_type(self, anchor_type):
        """Switch anchor type mid-stream; resets smoother state since old
        filter history is meaningless for a different anchor shape."""
        if anchor_type != self.anchor_type:
            self.anchor_type = anchor_type
            self.smoother.reset()
            self.last_resolved_type = None

    def process_frame(self, frame_bgr):
        """Run one frame through the full pipeline.

        Returns ``(payload, keypoints, resolved, segmentation_mask)`` — 
        all four are ``None`` if no pose was detected this frame.
        """
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            self.frame_timestamp_ms += 33
            detection_result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)

            if not detection_result.pose_landmarks:
                return None, None, None, None

            landmarks = detection_result.pose_landmarks[0]
            h, w, _ = frame_bgr.shape
            keypoints = extract_keypoints(landmarks)

            resolved = resolve_anchor(self.anchor_type, keypoints)

            # Only reset smoother if the core resolved_type changes
            # (not just the fallback state toggling)
            current_resolved_type = resolved["resolved_type"]
            if current_resolved_type != self.last_resolved_type:
                self.smoother.reset()
                self.last_resolved_type = current_resolved_type

            raw_transform = compute_spatial_transform(resolved, keypoints, w, h)

            # Local lighting extraction based on resolved anchor points
            lighting = extract_lighting(frame_bgr, raw_transform["points"])
            raw_transform["brightness"] = lighting["brightness"]

            smooth_tf = self.smoother.smooth(raw_transform)

            curr_time = time.time()
            fps = 1 / max(curr_time - self.prev_time, 1e-5)
            self.prev_time = curr_time

            payload = build_overlay_payload(smooth_tf, lighting, resolved, fps)
            
            segmentation_mask = None
            if detection_result.segmentation_masks:
                segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
                
            return payload, keypoints, resolved, segmentation_mask

        except Exception as e:
            print(f"Pipeline Error: {e}")
            return None, None, None, None