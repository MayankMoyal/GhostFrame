"""Prop Processor — Simple, reliable pivot + face_wear orientation for the OBS overlay path."""
import cv2
import numpy as np
from typing import Tuple


def process_prop_image(image_path: str, anchor_type: str) -> Tuple[float, float]:
    """
    Returns (grip_x, grip_y) normalized pivot coordinates for the OBS overlay.
    
    For face_wear, also corrects upside-down masks/glasses using center-of-mass
    detection and overwrites the image on disk.
    """
    # Category-specific pivots that match the local engine's prop_config.py
    _PIVOTS = {
        "hand_held":  (0.5, 0.90),   # Grip near the bottom (handle of sword)
        "face_wear":  (0.5, 0.35),   # Eye-hole level (mask eyes align with user's eyes)
        "head_wear":  (0.5, 0.95),   # Bottom center (base of hat sits on head)
        "neck_wear":  (0.5, 0.05),   # Top center (necklace hangs from neck)
        "wrist_wear": (0.5, 0.5),    # Center
        "ear_wear":   (0.5, 0.05),   # Top center (earring hangs from ear)
        "body_wear":  (0.5, 0.05),   # Top center (cape drapes from shoulders)
        "shield":     (0.5, 0.5),    # Center
        "background": (0.5, 0.5),    # Not used, but safe default
    }

    # ── FACE_WEAR: fix upside-down masks/glasses using center of mass ──
    if anchor_type == "face_wear":
        rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.ndim >= 3 and rgba.shape[2] == 4:
            h, w = rgba.shape[:2]
            alpha = rgba[:, :, 3]
            moments = cv2.moments(alpha)
            if moments["m00"] > 0:
                cy_ratio = (moments["m01"] / moments["m00"]) / h
                if cy_ratio > 0.55:
                    # Center of mass is below center → upside down → flip
                    rgba = cv2.flip(rgba, 0)
                    cv2.imwrite(image_path, rgba)
                    print(f"[prop_processor] Flipped FACE_WEAR (center of mass was at {cy_ratio:.0%})")
        return _PIVOTS["face_wear"]
    
    return _PIVOTS.get(anchor_type, (0.5, 0.5))
