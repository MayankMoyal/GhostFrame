"""Live Webcam -> Universal Prop Attachment Engine -> Display"""
import argparse
import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from prop_config import (
    AttachmentProfile, PropCategory, PROFILES, PropRequest,
    load_pipeline_json, resolve_category, classify_from_prompt,
)
from prop_manager import PropManager, PropMetadata
from tracker import UltimateTracker, OneEuroFilter, TrackResult

class _ScaleFilter:
    def __init__(self, initial: float = 1.0):
        self._f = OneEuroFilter(x0=initial, min_cutoff=0.1, beta=0.05)
    def __call__(self, x: float) -> float:
        return self._f(x)

class _PropState:
    GRACE_SECONDS = 0.20
    FADE_SECONDS  = 0.15
    def __init__(self):
        self.anchor = None
        self.rotation = 0.0
        self.scale = 0.5
        self.prop_img = None
        self.pivot = None
        self.last_seen = 0.0
        self.active = False

    def update(self, anchor, rotation, scale, prop_img, pivot):
        self.anchor, self.rotation, self.scale, self.prop_img, self.pivot = anchor, rotation, scale, prop_img, pivot
        self.last_seen = time.time()
        self.active = True

    def get_fallback(self):
        if not self.active or self.anchor is None: return None
        elapsed = time.time() - self.last_seen
        if elapsed > self.GRACE_SECONDS:
            self.active = False
            return None
        fade_start = self.GRACE_SECONDS - self.FADE_SECONDS
        opacity = 1.0 if elapsed <= fade_start else 1.0 - (elapsed - fade_start) / self.FADE_SECONDS
        return (self.anchor, self.rotation, self.scale, self.prop_img, self.pivot, max(0.0, min(1.0, opacity)))

# FAST OVERLAY (NO MEMORY LEAK, NO CUTTING)
def overlay_prop(frame, prop_rgba, target_xy, pivot, angle_deg, scale_factor, antialias=True, debug=False):
    h_frame, w_frame = frame.shape[:2]
    ph, pw = prop_rgba.shape[:2]
    new_w, new_h = int(pw * scale_factor), int(ph * scale_factor)
    if new_w <= 0 or new_h <= 0: return frame

    # Cap size to prevent memory issues
    max_w, max_h = int(w_frame * 0.80), int(h_frame * 0.80)
    if new_w > max_w or new_h > max_h:
        shrink = min(max_w / new_w, max_h / new_h)
        new_w, new_h = int(new_w * shrink), int(new_h * shrink)
    new_w, new_h = max(10, new_w), max(10, new_h)

    prop_scaled = cv2.resize(prop_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 1. TRUE rotated bounding box, computed from the actual rotated corners
    #    relative to the pivot. This matters because pivots are usually NOT
    #    at the image center (e.g. a sword's pivot is near the handle, ~85%
    #    down) — the old "w*cos+h*sin, centered on pivot" shortcut silently
    #    assumed a centered pivot and cropped off whatever stuck out further
    #    on one side (e.g. the entire blade tip).
    pvt_x, pvt_y = new_w * pivot[0], new_h * pivot[1]
    rad = math.radians(angle_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    corners = [(0, 0), (new_w, 0), (0, new_h), (new_w, new_h)]
    rot_xs, rot_ys = [], []
    for cx, cy in corners:
        dx, dy = cx - pvt_x, cy - pvt_y
        # matches OpenCV's own rotation convention (getRotationMatrix2D)
        rot_xs.append(dx * cos_r + dy * sin_r)
        rot_ys.append(-dx * sin_r + dy * cos_r)
    min_x, max_x = min(rot_xs), max(rot_xs)
    min_y, max_y = min(rot_ys), max(rot_ys)
    bw = max(10, int(math.ceil(max_x - min_x)))
    bh = max(10, int(math.ceil(max_y - min_y)))

    # 2. Where the pivot actually lands inside that (bw, bh) canvas —
    #    NOT necessarily the center.
    pivot_cx, pivot_cy = -min_x, -min_y
    M = cv2.getRotationMatrix2D((pvt_x, pvt_y), angle_deg, 1.0)
    M[0, 2] += pivot_cx - pvt_x
    M[1, 2] += pivot_cy - pvt_y

    rotated = cv2.warpAffine(prop_scaled, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    x1, y1 = int(target_xy[0] - pivot_cx), int(target_xy[1] - pivot_cy)
    x2, y2 = x1 + bw, y1 + bh
    
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w_frame, x2), min(h_frame, y2)
    if cx2 <= cx1 or cy2 <= cy1: return frame

    sx1, sy1 = cx1 - x1, cy1 - y1
    sx2, sy2 = sx1 + (cx2 - cx1), sy1 + (cy2 - cy1)
    roi = frame[cy1:cy2, cx1:cx2]
    prop_roi = rotated[sy1:sy2, sx1:sx2]
    alpha = prop_roi[:, :, 3].astype(np.float32) / 255.0
    if antialias and alpha.shape[0] > 3 and alpha.shape[1] > 3:
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)
    a3 = alpha[..., None]
    frame[cy1:cy2, cx1:cx2] = (
        prop_roi[:, :, :3].astype(np.float32) * a3 + roi.astype(np.float32) * (1.0 - a3)
    ).astype(np.uint8)
    return frame

def _to_px(norm_xy, mirror, w, h):
    x, y = norm_xy
    return (int((1.0 - x) * w), int(y * h)) if mirror else (int(x * w), int(y * h))

def _select_hand(hands, target_side):
    if not hands: return None
    if target_side == "any": return hands[0]
    label = target_side.capitalize()
    for hnd in hands:
        if hnd.handedness == label: return hnd
    return None

def _anchor_hand_held(track, meta, mirror, w, h):
    hand = _select_hand(track.hands, meta.target_side)
    if hand is None or hand.confidence < 0.3: return None
    return _to_px(hand.palm, mirror, w, h)

def _anchor_shield(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None: return None
    side = meta.target_side if meta.target_side != "any" else "left"
    wrist = pose.left_wrist_pose if side == "left" else pose.right_wrist_pose
    elbow = pose.left_elbow if side == "left" else pose.right_elbow
    if wrist is None or elbow is None: return None
    return _to_px(((wrist[0] + elbow[0]) / 2, (wrist[1] + elbow[1]) / 2), mirror, w, h)

def _anchor_head_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None or pose.head_center is None: return None
    ed = pose.ear_distance or 0
    return _to_px((pose.head_center[0], pose.head_center[1] - ed * 0.5), mirror, w, h)

def _anchor_neck_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None or pose.neck_point is None: return None
    return _to_px(pose.neck_point, mirror, w, h)

def _anchor_wrist_wear(track, meta, mirror, w, h):
    hand = _select_hand(track.hands, meta.target_side)
    if hand is not None: return _to_px(hand.wrist, mirror, w, h)
    pose = track.pose
    if pose is None: return None
    side = meta.target_side if meta.target_side != "any" else "right"
    wrist = pose.left_wrist_pose if side == "left" else pose.right_wrist_pose
    return _to_px(wrist, mirror, w, h) if wrist else None

def _anchor_ear_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None: return None
    side = meta.target_side if meta.target_side != "any" else "left"
    ear = pose.left_ear if side == "left" else pose.right_ear
    return _to_px(ear, mirror, w, h) if ear else None

def _anchor_face_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None: return None
    if pose.left_eye and pose.right_eye:
        return _to_px(((pose.left_eye[0] + pose.right_eye[0]) / 2, (pose.left_eye[1] + pose.right_eye[1]) / 2), mirror, w, h)
    return _to_px(pose.nose, mirror, w, h) if pose.nose else None

def _anchor_body_wear(track, meta, mirror, w, h):
    pose = track.pose
    if pose is None or pose.left_shoulder is None or pose.right_shoulder is None: return None
    return _to_px(((pose.left_shoulder[0] + pose.right_shoulder[0]) / 2, (pose.left_shoulder[1] + pose.right_shoulder[1]) / 2), mirror, w, h)

_ANCHOR_FN = {
    PropCategory.HAND_HELD: _anchor_hand_held, PropCategory.SHIELD: _anchor_shield,
    PropCategory.HEAD_WEAR: _anchor_head_wear, PropCategory.NECK_WEAR: _anchor_neck_wear,
    PropCategory.WRIST_WEAR: _anchor_wrist_wear, PropCategory.EAR_WEAR: _anchor_ear_wear,
    PropCategory.FACE_WEAR: _anchor_face_wear, PropCategory.BODY_WEAR: _anchor_body_wear,
}

def _compute_rotation(track, profile, meta, mirror):
    mode = profile.rotation_mode
    if mode == "hand_vector":
        hand = _select_hand(track.hands, meta.target_side)
        if hand is None: return 0.0
        angle_rad = math.pi - hand.hand_angle if mirror else hand.hand_angle
        base_deg = -(math.degrees(angle_rad) + 90)
        return base_deg + 80 if hand.handedness == "Right" else base_deg - 80
    if mode == "head_tilt":
        pose = track.pose
        return -pose.head_angle if pose and pose.head_angle is not None and mirror else (pose.head_angle if pose else 0.0)
    if mode == "shoulder_tilt":
        pose = track.pose
        return -pose.shoulder_angle if pose and pose.shoulder_angle is not None and mirror else (pose.shoulder_angle if pose else 0.0)
    if mode == "forearm_angle":
        pose = track.pose
        if pose is None: return 0.0
        side = meta.target_side if meta.target_side != "any" else "left"
        elbow, wrist = (pose.left_elbow, pose.left_wrist_pose) if side == "left" else (pose.right_elbow, pose.right_wrist_pose)
        if elbow and wrist:
            angle_rad = math.atan2(wrist[1] - elbow[1], wrist[0] - elbow[0])
            if mirror: angle_rad = math.pi - angle_rad
            return -(math.degrees(angle_rad) + 90)
        return 0.0
    return 0.0

def _get_body_measurement_px(track, profile, meta, w, h):
    ref = profile.body_scale_ref
    if ref == "palm_width":
        hand = _select_hand(track.hands, meta.target_side)
        if hand is None: return None
        ix, iy = hand.index_mcp[0] * w, hand.index_mcp[1] * h
        kx, ky = hand.pinky_mcp[0] * w, hand.pinky_mcp[1] * h
        return math.dist((ix, iy), (kx, ky))
    pose = track.pose
    if pose is None: return None
    if ref == "ear_distance": return pose.ear_distance * w if pose.ear_distance else None
    if ref == "shoulder_width": return pose.shoulder_width * w if pose.shoulder_width else None
    if ref == "forearm_length":
        side = meta.target_side if meta.target_side != "any" else "left"
        fl = pose.left_forearm_length if side == "left" else pose.right_forearm_length
        return fl * h if fl else None
    if ref == "ear_eye_distance":
        side = meta.target_side if meta.target_side != "any" else "left"
        d = pose.left_ear_eye_dist if side == "left" else pose.right_ear_eye_dist
        return d * w if d else None
    if ref == "torso_height": return pose.torso_height * h if pose.torso_height else None
    return None

def _get_prop_measurement_px(meta, profile):
    ref = profile.prop_scale_ref
    if ref == "handle_width": return meta.handle_width
    if ref == "width": return meta.image.shape[1]
    if ref == "height": return meta.image.shape[0]
    return meta.image.shape[1]

def _needs_flip(profile, meta, track):
    cat = profile.category
    if cat in (PropCategory.HAND_HELD, PropCategory.WRIST_WEAR):
        hand = _select_hand(track.hands, meta.target_side)
        if hand and hand.handedness == "Left": return True
    if cat == PropCategory.SHIELD: return (meta.target_side if meta.target_side != "any" else "left") == "right"
    if cat == PropCategory.EAR_WEAR: return (meta.target_side if meta.target_side != "any" else "left") == "right"
    return False

def render_prop(display, pdata, track, mirror, w, h, debug):
    meta = pdata["meta"]
    profile = meta.profile
    state = pdata["state"]
    anchor_fn = _ANCHOR_FN.get(profile.category)
    if anchor_fn is None: return display
    anchor = anchor_fn(track, meta, mirror, w, h)
    opacity = 1.0

    if anchor is not None:
        rotation = _compute_rotation(track, profile, meta, mirror)
        body_px = _get_body_measurement_px(track, profile, meta, w, h)
        prop_px = _get_prop_measurement_px(meta, profile)
        
        if body_px and prop_px and prop_px > 0:
            raw_scale = (body_px / prop_px) * profile.scale_multiplier
            
            # --- DYNAMIC SAFETY CLAMP ---
            # Calculate the max scale allowed so the prop never cuts off the screen
            ph, pw = meta.image.shape[:2]
            max_scale_h = (h * 0.75) / ph
            max_scale_w = (w * 0.60) / pw
            max_safe_scale = min(max_scale_h, max_scale_w)
            raw_scale = min(raw_scale, max_safe_scale)
        else:
            raw_scale = 0.5
            
        scale = pdata["scale_filter"](raw_scale)
        prop_img = meta.image_flipped if _needs_flip(profile, meta, track) else meta.image
        pivot = meta.pivot_flipped if _needs_flip(profile, meta, track) else meta.pivot
        state.update(anchor, rotation, scale, prop_img, pivot)
    else:
        fallback = state.get_fallback()
        if fallback is None: return display
        anchor, rotation, scale, prop_img, pivot, opacity = fallback

    if opacity < 1.0:
        temp = display.copy()
        temp = overlay_prop(temp, prop_img, anchor, pivot, rotation, scale, debug=debug)
        cv2.addWeighted(temp, opacity, display, 1.0 - opacity, 0, display)
    else:
        display = overlay_prop(display, prop_img, anchor, pivot, rotation, scale, debug=debug)

    if debug:
        cv2.circle(display, anchor, 6, (0, 255, 0), -1)
        cv2.putText(display, f"{profile.category.value} ({opacity:.0%})", (anchor[0] - 40, anchor[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return display

def main():
    ap = argparse.ArgumentParser(description="Universal Prop Attachment Engine")
    ap.add_argument("--props", nargs="+", help="Prop images. Format: path.png or path.png:category")
    ap.add_argument("--from-pipeline", dest="pipeline_json", help="JSON file with PropRequest list")
    ap.add_argument("--hand", choices=["left", "right", "any"], default="any")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    if not args.props and not args.pipeline_json: ap.error("Provide --props or --from-pipeline")
    active_props = []

    if args.pipeline_json:
        for req in load_pipeline_json(args.pipeline_json):
            cat = resolve_category(req.category)
            side = req.target_side or args.hand
            meta = PropManager.process_image(req.image_path, category=cat, prompt=req.prompt, target_side=side)
            if meta: active_props.append({"meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()})

    if args.props:
        for spec in args.props:
            if ":" in spec:
                path, cat_str = spec.rsplit(":", 1)
                cat = resolve_category(cat_str)
            else:
                path, cat = spec, None
            meta = PropManager.process_image(path, category=cat, target_side=args.hand)
            if meta: active_props.append({"meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()})

    if not active_props:
        print("[ERROR] No props loaded. Exiting."); return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = UltimateTracker()
    mirror = not args.no_mirror
    print(f"\nTracking {len(active_props)} prop(s) ... press ESC to quit.\n")
    fps_t, fps_n, fps_val = time.time(), 0, 0.0

    while True:
        ok, raw = cap.read()
        if not ok: break
        
        h, w = raw.shape[:2]
        result = tracker.detect(raw)
        display = cv2.flip(raw, 1) if mirror else raw.copy()

        for pdata in active_props:
            display = render_prop(display, pdata, result, mirror, w, h, args.debug)

        if not result.hands:
            cv2.putText(display, "SHOW YOUR HAND TO CAMERA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        fps_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            fps_val = fps_n / elapsed; fps_n = 0; fps_t = time.time()
        cv2.putText(display, f"FPS: {fps_val:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Prop Engine", display)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()