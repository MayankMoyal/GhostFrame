"""Live Webcam -> Universal Prop Attachment Engine -> Display

Supports 8 prop categories, each with profile-driven anchor, rotation,
and scale computation.  Adding a new category requires only a new
AttachmentProfile entry in prop_config.py -- zero rendering code changes.

Input modes:
    --props sword.png                       (backward compat, defaults to hand_held)
    --props sword.png:hand_held crown.png:head_wear
    --from-pipeline pipeline.json           (JSON PropRequest list)
"""
import argparse
import json
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from prop_config import (
    AttachmentProfile,
    PropCategory,
    PROFILES,
    PropRequest,
    load_pipeline_json,
    resolve_category,
    classify_from_prompt,
)
from prop_manager import PropManager, PropMetadata
from tracker import UltimateTracker, OneEuroFilter, TrackResult


# --- Scale smoother ---
class _ScaleFilter:
    def __init__(self, initial: float = 1.0):
        self._f = OneEuroFilter(x0=initial, min_cutoff=0.1, beta=0.05)

    def __call__(self, x: float) -> float:
        return self._f(x)


# --- Temporal persistence state ---
class _PropState:
    """Caches last known rendering params so the prop survives brief tracking drops.

    When the tracker loses the hand for a few frames (common during tilts),
    we keep rendering at the last known position with a fade-out instead of
    making the prop blink in and out.
    """
    GRACE_SECONDS = 0.20   # hold for 200 ms after tracking loss
    FADE_SECONDS  = 0.15   # fade out over the last 150 ms of the grace period

    def __init__(self):
        self.anchor = None
        self.rotation = 0.0
        self.scale = 0.5
        self.prop_img = None
        self.pivot = None
        self.last_seen = 0.0   # time.time() of last successful detection
        self.active = False

    def update(self, anchor, rotation, scale, prop_img, pivot):
        """Call on every frame where tracking succeeds."""
        self.anchor = anchor
        self.rotation = rotation
        self.scale = scale
        self.prop_img = prop_img
        self.pivot = pivot
        self.last_seen = time.time()
        self.active = True

    def get_fallback(self):
        """Return (anchor, rotation, scale, prop_img, pivot, opacity) or None.

        Returns cached values with fading opacity during the grace window.
        Returns None if the grace period has expired.
        """
        if not self.active or self.anchor is None:
            return None
        elapsed = time.time() - self.last_seen
        if elapsed > self.GRACE_SECONDS:
            self.active = False
            return None
        # Compute opacity: full for most of the grace period, fade near the end
        fade_start = self.GRACE_SECONDS - self.FADE_SECONDS
        if elapsed <= fade_start:
            opacity = 1.0
        else:
            opacity = 1.0 - (elapsed - fade_start) / self.FADE_SECONDS
        opacity = max(0.0, min(1.0, opacity))
        return (self.anchor, self.rotation, self.scale,
                self.prop_img, self.pivot, opacity)


# ===========================================================================
#  OVERLAY COMPOSITING (expanded canvas, AA alpha)
# ===========================================================================
def overlay_prop(frame, prop_rgba, target_xy, pivot, angle_deg, scale_factor,
                 antialias=True, debug=False):
    h_frame, w_frame = frame.shape[:2]
    ph, pw = prop_rgba.shape[:2]

    new_w = int(pw * scale_factor)
    new_h = int(ph * scale_factor)
    if new_w <= 0 or new_h <= 0:
        return frame

    max_w = int(w_frame * 0.55)
    max_h = int(h_frame * 0.90)
    if new_w > max_w or new_h > max_h:
        shrink = min(max_w / new_w, max_h / new_h)
        new_w = int(new_w * shrink)
        new_h = int(new_h * shrink)
    new_w, new_h = max(10, new_w), max(10, new_h)

    prop_scaled = cv2.resize(prop_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate exact max distance from pivot to any corner
    pvt_x_orig = new_w * pivot[0]
    pvt_y_orig = new_h * pivot[1]
    corners = [(0, 0), (new_w, 0), (0, new_h), (new_w, new_h)]
    max_dist = max(math.dist((pvt_x_orig, pvt_y_orig), c) for c in corners)

    # Canvas size needs to be 2 * max_dist to allow rotation around pivot without clipping
    canvas_size = int(math.ceil(max_dist * 2))
    canvas_size += canvas_size % 2  # make it even

    # Place the prop so that its pivot is EXACTLY at the center of the canvas
    pad_x = int(canvas_size / 2 - pvt_x_orig)
    pad_y = int(canvas_size / 2 - pvt_y_orig)

    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = prop_scaled

    pvt_cx = canvas_size / 2
    pvt_cy = canvas_size / 2

    M = cv2.getRotationMatrix2D((pvt_cx, pvt_cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(canvas, M, (canvas_size, canvas_size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    x1 = int(target_xy[0] - pvt_cx)
    y1 = int(target_xy[1] - pvt_cy)
    x2, y2 = x1 + canvas_size, y1 + canvas_size
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w_frame, x2), min(h_frame, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return frame

    sx1, sy1 = cx1 - x1, cy1 - y1
    sx2, sy2 = sx1 + (cx2 - cx1), sy1 + (cy2 - cy1)
    roi = frame[cy1:cy2, cx1:cx2]
    prop_roi = rotated[sy1:sy2, sx1:sx2]
    alpha = prop_roi[:, :, 3].astype(np.float32) / 255.0
    if antialias and alpha.shape[0] > 3 and alpha.shape[1] > 3:
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)
    a3 = alpha[..., None]
    frame[cy1:cy2, cx1:cx2] = (
        prop_roi[:, :, :3].astype(np.float32) * a3
        + roi.astype(np.float32) * (1.0 - a3)
    ).astype(np.uint8)
    return frame


# ===========================================================================
#  DISPATCHER FUNCTIONS -- profile-driven anchor / rotation / scale
# ===========================================================================

def _to_px(norm_xy, mirror, w, h):
    x, y = norm_xy
    if mirror:
        return (int((1.0 - x) * w), int(y * h))
    return (int(x * w), int(y * h))


def _select_hand(hands, target_side):
    if not hands:
        return None
    if target_side == "any":
        return hands[0]
    label = target_side.capitalize()
    for hnd in hands:
        if hnd.handedness == label:
            return hnd
    return None


# -- Anchor computation --
def _anchor_hand_held(track, meta, mirror, w, h):
    hand = _select_hand(track.hands, meta.target_side)
    if hand is None or hand.confidence < 0.3:
        return None
    return _to_px(hand.palm, mirror, w, h)


def _anchor_shield(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None:
        return None
    side = meta.target_side if meta.target_side != "any" else "left"
    wrist = pose.left_wrist_pose if side == "left" else pose.right_wrist_pose
    elbow = pose.left_elbow if side == "left" else pose.right_elbow
    if wrist is None or elbow is None:
        return None
    mid = ((wrist[0] + elbow[0]) / 2, (wrist[1] + elbow[1]) / 2)
    return _to_px(mid, mirror, w, h)


def _anchor_head_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None or pose.head_center is None:
        return None
    hc = pose.head_center
    ed = pose.ear_distance or 0
    anchor = (hc[0], hc[1] - ed * 0.5)
    return _to_px(anchor, mirror, w, h)


def _anchor_neck_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None or pose.neck_point is None:
        return None
    return _to_px(pose.neck_point, mirror, w, h)


def _anchor_wrist_wear(track, meta, mirror, w, h):
    hand = _select_hand(track.hands, meta.target_side)
    if hand is not None:
        return _to_px(hand.wrist, mirror, w, h)
    pose = track.pose
    if pose is None:
        return None
    side = meta.target_side if meta.target_side != "any" else "right"
    wrist = pose.left_wrist_pose if side == "left" else pose.right_wrist_pose
    if wrist is None:
        return None
    return _to_px(wrist, mirror, w, h)


def _anchor_ear_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None:
        return None
    side = meta.target_side if meta.target_side != "any" else "left"
    ear = pose.left_ear if side == "left" else pose.right_ear
    if ear is None:
        return None
    return _to_px(ear, mirror, w, h)


def _anchor_face_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None:
        return None
    # Prefer midpoint of eyes (glasses bridge); fall back to nose
    if pose.left_eye and pose.right_eye:
        mid = ((pose.left_eye[0] + pose.right_eye[0]) / 2,
               (pose.left_eye[1] + pose.right_eye[1]) / 2)
        return _to_px(mid, mirror, w, h)
    if pose.nose:
        return _to_px(pose.nose, mirror, w, h)
    return None


def _anchor_body_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None or pose.left_shoulder is None or pose.right_shoulder is None:
        return None
    mid = ((pose.left_shoulder[0] + pose.right_shoulder[0]) / 2,
           (pose.left_shoulder[1] + pose.right_shoulder[1]) / 2)
    return _to_px(mid, mirror, w, h)


_ANCHOR_FN = {
    PropCategory.HAND_HELD:  _anchor_hand_held,
    PropCategory.SHIELD:     _anchor_shield,
    PropCategory.HEAD_WEAR:  _anchor_head_wear,
    PropCategory.NECK_WEAR:  _anchor_neck_wear,
    PropCategory.WRIST_WEAR: _anchor_wrist_wear,
    PropCategory.EAR_WEAR:   _anchor_ear_wear,
    PropCategory.FACE_WEAR:  _anchor_face_wear,
    PropCategory.BODY_WEAR:  _anchor_body_wear,
}


# -- Rotation computation --
def _compute_rotation(track, profile, meta, mirror):
    mode = profile.rotation_mode

    if mode == "hand_vector":
        hand = _select_hand(track.hands, meta.target_side)
        if hand is None:
            return 0.0
        angle_rad = hand.hand_angle
        if mirror:
            angle_rad = math.pi - angle_rad
        
        base_deg = -(math.degrees(angle_rad) + 90)
        
        # Offset by 80 degrees so the prop sits across the palm (like a grip).
        # We want the blade to point in the direction of the thumb.
        # Right hand thumb is on the left (w/ mirror) -> +80 deg (CCW)
        if hand.handedness == "Right":
            return base_deg + 80
        else:
            return base_deg - 80

    if mode == "head_tilt":
        pose = track.pose
        if pose and pose.head_angle is not None:
            return -pose.head_angle if mirror else pose.head_angle
        return 0.0

    if mode == "shoulder_tilt":
        pose = track.pose
        if pose and pose.shoulder_angle is not None:
            return -pose.shoulder_angle if mirror else pose.shoulder_angle
        return 0.0

    if mode == "forearm_angle":
        pose = track.pose
        if pose is None:
            return 0.0
        side = meta.target_side if meta.target_side != "any" else "left"
        if side == "left":
            elbow, wrist = pose.left_elbow, pose.left_wrist_pose
        else:
            elbow, wrist = pose.right_elbow, pose.right_wrist_pose
        if elbow and wrist:
            dx = wrist[0] - elbow[0]
            dy = wrist[1] - elbow[1]
            angle_rad = math.atan2(dy, dx)
            if mirror:
                angle_rad = math.pi - angle_rad
            return -(math.degrees(angle_rad) + 90)
        return 0.0

    return 0.0  # "none"


# -- Scale computation --
def _get_body_measurement_px(track, profile, meta, w, h):
    """Return the body measurement in pixels for scale calibration."""
    ref = profile.body_scale_ref

    if ref == "palm_width":
        hand = _select_hand(track.hands, meta.target_side)
        if hand is None:
            return None
        ix, iy = hand.index_mcp[0] * w, hand.index_mcp[1] * h
        kx, ky = hand.pinky_mcp[0] * w, hand.pinky_mcp[1] * h
        return math.dist((ix, iy), (kx, ky))

    pose = track.pose
    if pose is None:
        return None

    if ref == "ear_distance":
        return pose.ear_distance * w if pose.ear_distance else None

    if ref == "shoulder_width":
        return pose.shoulder_width * w if pose.shoulder_width else None

    if ref == "forearm_length":
        side = meta.target_side if meta.target_side != "any" else "left"
        fl = (pose.left_forearm_length if side == "left"
              else pose.right_forearm_length)
        return fl * h if fl else None  # forearm is mostly vertical

    if ref == "ear_eye_distance":
        side = meta.target_side if meta.target_side != "any" else "left"
        d = (pose.left_ear_eye_dist if side == "left"
             else pose.right_ear_eye_dist)
        return d * w if d else None

    if ref == "torso_height":
        return pose.torso_height * h if pose.torso_height else None

    return None


def _get_prop_measurement_px(meta, profile):
    """Return the prop dimension in pixels to match against the body measurement."""
    ref = profile.prop_scale_ref
    if ref == "handle_width":
        return meta.handle_width
    if ref == "width":
        return meta.image.shape[1]
    if ref == "height":
        return meta.image.shape[0]
    return meta.image.shape[1]


# -- Prop image selection (flip for handedness / side) --
def _needs_flip(profile, meta, track):
    """Determine if the prop should be flipped for the opposite hand/side."""
    cat = profile.category
    if cat in (PropCategory.HAND_HELD, PropCategory.WRIST_WEAR):
        hand = _select_hand(track.hands, meta.target_side)
        if hand and hand.handedness == "Left":
            return True
    if cat == PropCategory.SHIELD:
        side = meta.target_side if meta.target_side != "any" else "left"
        return side == "right"
    if cat == PropCategory.EAR_WEAR:
        side = meta.target_side if meta.target_side != "any" else "left"
        return side == "right"
    return False


# ===========================================================================
#  UNIVERSAL RENDERER
# ===========================================================================
def render_prop(display, pdata, track, mirror, w, h, debug):
    """Profile-driven rendering with temporal persistence.

    If the tracker loses the hand/body part momentarily (common during
    tilts), we keep rendering at the last known position with a fade-out
    instead of making the prop blink in and out.
    """
    meta = pdata["meta"]
    profile = meta.profile
    state = pdata["state"]    # _PropState for temporal persistence

    # 1. Try to get a live anchor
    anchor_fn = _ANCHOR_FN.get(profile.category)
    if anchor_fn is None:
        return display
    anchor = anchor_fn(track, meta, mirror, w, h)

    opacity = 1.0

    if anchor is not None:
        # --- Live tracking available ---
        rotation = _compute_rotation(track, profile, meta, mirror)

        body_px = _get_body_measurement_px(track, profile, meta, w, h)
        prop_px = _get_prop_measurement_px(meta, profile)
        if body_px and prop_px and prop_px > 0:
            raw_scale = (body_px / prop_px) * profile.scale_multiplier
        else:
            raw_scale = 0.5
        scale = pdata["scale_filter"](raw_scale)

        if _needs_flip(profile, meta, track):
            prop_img = meta.image_flipped
            pivot = meta.pivot_flipped
        else:
            prop_img = meta.image
            pivot = meta.pivot

        # Cache for grace period
        state.update(anchor, rotation, scale, prop_img, pivot)

    else:
        # --- Tracking lost: use cached state during grace period ---
        fallback = state.get_fallback()
        if fallback is None:
            return display  # grace expired, nothing to render
        anchor, rotation, scale, prop_img, pivot, opacity = fallback

    # Composite (with opacity if fading out)
    if opacity < 1.0:
        # Render to a temp buffer and blend with reduced opacity
        temp = display.copy()
        temp = overlay_prop(temp, prop_img, anchor, pivot, rotation, scale,
                            debug=debug)
        cv2.addWeighted(temp, opacity, display, 1.0 - opacity, 0, display)
    else:
        display = overlay_prop(display, prop_img, anchor, pivot, rotation, scale,
                               debug=debug)

    # Debug overlay
    if debug:
        cv2.circle(display, anchor, 6, (0, 255, 0), -1)
        label = f"{profile.category.value} ({opacity:.0%})"
        cv2.putText(display, label,
                    (anchor[0] - 40, anchor[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1, cv2.LINE_AA)

    return display


# ===========================================================================
#  MAIN LOOP
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Universal Prop Attachment Engine")
    ap.add_argument("--props", nargs="+",
                    help="Prop images.  Format: path.png  or  path.png:category")
    ap.add_argument("--from-pipeline", dest="pipeline_json",
                    help="JSON file with PropRequest list from image-gen pipeline")
    ap.add_argument("--hand", choices=["left", "right", "any"], default="any",
                    help="Default hand/side for props (overridable per-prop in JSON)")
    ap.add_argument("--debug", action="store_true",
                    help="Show landmark dots and per-frame diagnostics")
    ap.add_argument("--no-mirror", action="store_true",
                    help="Disable selfie mirror")
    args = ap.parse_args()

    if not args.props and not args.pipeline_json:
        ap.error("Provide --props or --from-pipeline")

    # -- Build prop list --
    active_props = []

    if args.pipeline_json:
        # JSON pipeline input
        requests = load_pipeline_json(args.pipeline_json)
        for req in requests:
            cat = resolve_category(req.category)
            side = req.target_side or args.hand
            meta = PropManager.process_image(
                req.image_path, category=cat, prompt=req.prompt,
                target_side=side)
            if meta:
                active_props.append({"meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()})

    if args.props:
        # CLI props -- support path.png or path.png:category
        for spec in args.props:
            if ":" in spec:
                path, cat_str = spec.rsplit(":", 1)
                cat = resolve_category(cat_str)
            else:
                path = spec
                cat = None  # will fallback to HAND_HELD inside process_image
            meta = PropManager.process_image(
                path, category=cat, target_side=args.hand)
            if meta:
                active_props.append({"meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()})

    if not active_props:
        print("[ERROR] No props loaded. Exiting.")
        return

    # -- Camera --
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = UltimateTracker()
    mirror = not args.no_mirror

    print(f"\nTracking {len(active_props)} prop(s) ... press ESC to quit.\n")

    fps_t, fps_n, fps_val = time.time(), 0, 0.0

    while True:
        ok, raw = cap.read()
        if not ok:
            break
        h, w = raw.shape[:2]

        result = tracker.detect(raw)
        display = cv2.flip(raw, 1) if mirror else raw.copy()

        for pdata in active_props:
            display = render_prop(display, pdata, result, mirror, w, h, args.debug)

        # FPS
        fps_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            fps_val = fps_n / elapsed
            fps_n = 0
            fps_t = time.time()
        cv2.putText(display, f"FPS: {fps_val:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Prop Engine", display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
