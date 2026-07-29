"""Anchor resolution and spatial transform for the overlay system.

Given a requested anchor type (e.g. ``"right_wrist"``, ``"head"``,
``"both_shoulders"``), ``resolve_anchor`` maps it to one or two
concrete keypoints extracted from MediaPipe Pose, with a configurable
kinematic fallback chain (e.g. wrist → elbow → shoulder) so the
overlay degrades gracefully instead of jumping to ambient when a
single joint is momentarily occluded.

``compute_spatial_transform`` then converts the resolved anchor(s)
into pixel coordinates, a rotation angle, and a depth-aware scale
factor suitable for positioning an overlay asset on screen.
"""

import math

VISIBILITY_THRESHOLD = 0.5
DEFAULT_SHOULDER_DIST = 0.25

Z_DEPTH_BASELINE = 0.0
Z_DEPTH_SENSITIVITY = 2.0
Z_DEPTH_WEIGHT = 0.4
SHOULDER_DEPTH_WEIGHT = 0.6

ANCHOR_CONFIG = {
    "prop_in_hand":    {"strategy": "single", "candidates": ["right_wrist", "left_wrist"], "fallback": ["right_elbow", "left_elbow", "right_shoulder", "left_shoulder"]},
    "left_wrist":      {"strategy": "single", "candidates": ["left_wrist"], "fallback": ["left_elbow", "left_shoulder"]},
    "right_wrist":     {"strategy": "single", "candidates": ["right_wrist"], "fallback": ["right_elbow", "right_shoulder"]},
    "both_wrists":     {"strategy": "dual",   "candidates": ["left_wrist", "right_wrist"]},

    "head":            {"strategy": "single", "candidates": ["nose"]},
    "left_shoulder":   {"strategy": "single", "candidates": ["left_shoulder"]},
    "right_shoulder":  {"strategy": "single", "candidates": ["right_shoulder"]},
    "both_shoulders":  {"strategy": "dual",   "candidates": ["left_shoulder", "right_shoulder"]},

    "ambient":         {"strategy": "none", "candidates": []},
    "background":      {"strategy": "none", "candidates": []},
}


def resolve_anchor(requested_type, keypoints):
    """Resolve the requested anchor type to concrete keypoint(s).

    For single-point anchors, picks the most visible candidate from
    the primary list.  If none are visible, walks up the kinematic
    fallback chain (e.g. wrist → elbow → shoulder) before falling
    back to ambient.

    For dual-point anchors (``both_wrists``, ``both_shoulders``),
    both candidates must be visible; otherwise falls back to ambient.

    Parameters
    ----------
    requested_type : str
        One of the keys in ``ANCHOR_CONFIG``.
    keypoints : dict
        Keypoint dict from ``extract_keypoints()``.

    Returns
    -------
    dict
        ``resolved_type``, ``anchor_points`` (list of 0-2 dicts with
        x, y, z, source, visibility), ``fallback_applied`` (bool).
    """
    if requested_type not in ANCHOR_CONFIG:
        raise ValueError(f"Unknown anchor type: {requested_type}")

    config = ANCHOR_CONFIG[requested_type]

    if config["strategy"] == "none":
        return {"resolved_type": requested_type, "anchor_points": [], "fallback_applied": False}

    if config["strategy"] == "single":
        candidates = []
        # Check primary candidates
        for kp_name in config["candidates"]:
            kp = keypoints.get(kp_name)
            if kp and kp["visibility"] >= VISIBILITY_THRESHOLD:
                candidates.append({"x": kp["x"], "y": kp["y"], "z": kp.get("z", 0.0),
                                   "source": kp_name, "visibility": kp["visibility"]})
        if candidates:
            best = max(candidates, key=lambda c: c["visibility"])
            return {"resolved_type": requested_type, "anchor_points": [best], "fallback_applied": False}

        # Check fallback candidates (kinematic chain: wrist → elbow → shoulder)
        for kp_name in config.get("fallback", []):
            kp = keypoints.get(kp_name)
            if kp and kp["visibility"] >= VISIBILITY_THRESHOLD:
                candidates.append({"x": kp["x"], "y": kp["y"], "z": kp.get("z", 0.0),
                                   "source": kp_name, "visibility": kp["visibility"]})
        if candidates:
            best = max(candidates, key=lambda c: c["visibility"])
            # Keep resolved_type as original so pipeline doesn't reset smoother
            return {"resolved_type": requested_type, "anchor_points": [best], "fallback_applied": True}

        return {"resolved_type": "ambient", "anchor_points": [], "fallback_applied": True}

    if config["strategy"] == "dual":
        points = []
        for kp_name in config["candidates"]:
            kp = keypoints.get(kp_name)
            if kp and kp["visibility"] >= VISIBILITY_THRESHOLD:
                points.append({"x": kp["x"], "y": kp["y"], "z": kp.get("z", 0.0),
                               "source": kp_name, "visibility": kp["visibility"]})
        if len(points) == len(config["candidates"]):
            return {"resolved_type": requested_type, "anchor_points": points, "fallback_applied": False}
        return {"resolved_type": "ambient", "anchor_points": [], "fallback_applied": True}


def _compute_body_scale(keypoints):
    """Estimate how close the user is to the camera via body proportions.

    Uses the 2D shoulder width (in normalized coords) as the primary
    metric. Falls back to ear-to-ear distance if shoulders are off-screen.

    Note: we intentionally use 2D distance here, not 3D. MediaPipe's Z
    values are in a different unit/scale than X,Y (meters vs 0-1
    normalized), so mixing them in a Euclidean distance is incorrect.
    Z-depth is handled separately via ``_compute_z_scale``.

    Returns a multiplier in [0.2, 4.0] where 1.0 ≈ the user at a
    "default" distance.
    """
    ls = keypoints.get("left_shoulder")
    rs = keypoints.get("right_shoulder")

    # Primary: shoulder width (2D, normalized coordinates)
    if (ls and rs
            and ls["visibility"] >= VISIBILITY_THRESHOLD
            and rs["visibility"] >= VISIBILITY_THRESHOLD):
        current_dist = math.hypot(ls["x"] - rs["x"], ls["y"] - rs["y"])
        return max(0.2, min(4.0, current_dist / DEFAULT_SHOULDER_DIST))

    # Fallback: face width (ear-to-ear, roughly 1/3 of shoulder width)
    l_ear = keypoints.get("left_ear")
    r_ear = keypoints.get("right_ear")
    if (l_ear and r_ear
            and l_ear["visibility"] >= VISIBILITY_THRESHOLD
            and r_ear["visibility"] >= VISIBILITY_THRESHOLD):
        current_dist = math.hypot(l_ear["x"] - r_ear["x"], l_ear["y"] - r_ear["y"])
        DEFAULT_EAR_DIST = DEFAULT_SHOULDER_DIST / 3.0
        return max(0.2, min(4.0, current_dist / DEFAULT_EAR_DIST))

    return 1.0


def _compute_z_scale(z_value):
    """Convert MediaPipe's hip-relative Z depth into a scale multiplier."""
    delta = Z_DEPTH_BASELINE - z_value
    raw_scale = 1.0 + delta * Z_DEPTH_SENSITIVITY
    return max(0.5, min(2.5, raw_scale))


def _blend_scales(shoulder_scale, z_scale, has_z):
    """Weighted blend of body-proportion scale and Z-depth scale."""
    if not has_z:
        return shoulder_scale
    return SHOULDER_DEPTH_WEIGHT * shoulder_scale + Z_DEPTH_WEIGHT * z_scale


def compute_spatial_transform(resolved_data, keypoints, img_w, img_h):
    """Convert resolved anchor(s) into pixel coords, angle, and scale.

    Always returns a ``"points"`` list (1 entry for single/ambient,
    2 for dual anchors) so callers can handle both cases uniformly.
    Dual anchors return both raw pixel coordinates (no midpoint
    collapse) — useful for rendering spanning overlays like capes.

    Angle convention
    ----------------
    - **Wrist/elbow anchors**: ``atan2(dx, -dy)`` where the vector is
      elbow→wrist. This gives 0° when the arm points straight **up**
      and increases clockwise. A vertical sword image (blade up,
      handle down) can be rotated by this angle directly.
    - **Head (nose)**: tilt derived from ear-to-ear line, offset by
      -90° so a crown stays upright when the head is level.
    - **Dual anchors**: tilt of the line connecting the two points,
      using ``atan2(dx, -dy) + 90°``.

    Parameters
    ----------
    resolved_data : dict
        Output of ``resolve_anchor()``.
    keypoints : dict
        Full keypoint dict.
    img_w, img_h : int
        Frame dimensions in pixels.

    Returns
    -------
    dict
        ``points`` (list of {x, y}), ``angle`` (degrees), ``scale``.
    """
    body_scale = _compute_body_scale(keypoints)
    points = resolved_data["anchor_points"]

    # --- Ambient / no anchor: screen center ---
    if not points:
        return {
            "points": [{"x": round(img_w / 2.0, 1), "y": round(img_h / 2.0, 1)}],
            "angle": 0.0,
            "scale": round(body_scale, 2),
        }

    # --- Dual-point anchor: both points returned, no midpoint collapse ---
    if len(points) >= 2:
        p1, p2 = points[0], points[1]

        p1_px = {"x": round(p1["x"] * img_w, 1), "y": round(p1["y"] * img_h, 1)}
        p2_px = {"x": round(p2["x"] * img_w, 1), "y": round(p2["y"] * img_h, 1)}

        dx = (p2["x"] - p1["x"]) * img_w
        dy = (p2["y"] - p1["y"]) * img_h
        angle = math.degrees(math.atan2(dx, -dy)) + 90.0

        z1, z2 = p1.get("z", 0.0), p2.get("z", 0.0)
        has_z = (z1 != 0.0) or (z2 != 0.0)
        avg_z = (z1 + z2) / 2.0 if has_z else 0.0
        z_scale = _compute_z_scale(avg_z) if has_z else 1.0
        scale = _blend_scales(body_scale, z_scale, has_z)

        return {
            "points": [p1_px, p2_px],
            "angle": round(angle, 1),
            "scale": round(scale, 2),
        }

    # --- Single-point anchor: per-limb kinematics for angle ---
    pt = points[0]
    source = pt["source"]
    x_px = pt["x"] * img_w
    y_px = pt["y"] * img_h
    angle = 0.0

    anchor_z = pt.get("z", 0.0)
    has_z = anchor_z != 0.0
    z_scale = _compute_z_scale(anchor_z) if has_z else 1.0
    scale = _blend_scales(body_scale, z_scale, has_z)

    if "wrist" in source or "elbow" in source:
        # Angle from elbow→wrist vector: 0° = arm straight up, CW positive
        elbow_key = "left_elbow" if "left" in source else "right_elbow"
        elbow = keypoints.get(elbow_key)
        if elbow and elbow["visibility"] >= VISIBILITY_THRESHOLD:
            dx = (pt["x"] - elbow["x"]) * img_w
            dy = (pt["y"] - elbow["y"]) * img_h
            angle = math.degrees(math.atan2(dx, -dy))
        else:
            angle = 0.0

    elif source == "nose":
        l_ear = keypoints.get("left_ear")
        r_ear = keypoints.get("right_ear")
        if (l_ear and r_ear
                and l_ear["visibility"] >= VISIBILITY_THRESHOLD
                and r_ear["visibility"] >= VISIBILITY_THRESHOLD):
            dx = (r_ear["x"] - l_ear["x"]) * img_w
            dy = (r_ear["y"] - l_ear["y"]) * img_h
            # Offset so a crown stays vertical when head is level
            angle = math.degrees(math.atan2(dx, -dy)) - 90.0

    return {
        "points": [{"x": round(x_px, 1), "y": round(y_px, 1)}],
        "angle": round(angle, 1),
        "scale": round(scale, 2),
    }