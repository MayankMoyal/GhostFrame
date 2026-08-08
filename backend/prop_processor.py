"""Prop Processor — Simple, reliable orientation and pivot for the OBS overlay path."""
import cv2
import numpy as np
from typing import Tuple


def process_prop_image(image_path: str, anchor_type: str) -> Tuple[float, float]:
    """
    Returns (grip_x, grip_y) normalized pivot coordinates for the OBS overlay.
    
    Does NOT attempt complex PCA rotation or heuristic flipping — those were
    unreliable and caused swords to be grabbed by the blade and masks to appear
    upside down. Instead, we rely on the AI Agent prompt to generate images in
    the correct orientation, and return the appropriate fixed pivot per category.
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
    
    return _PIVOTS.get(anchor_type, (0.5, 0.5))
