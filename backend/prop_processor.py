import cv2
import numpy as np
import math
from typing import Tuple

def _avg_row_width(strip: np.ndarray, threshold: int = 127) -> float:
    widths = []
    for row in strip:
        cols = np.where(row > threshold)[0]
        if len(cols) > 0:
            widths.append(cols[-1] - cols[0])
    return float(np.mean(widths)) if widths else 0.0

def process_prop_image(image_path: str, anchor_type: str) -> Tuple[float, float]:
    """Rotates diagonal props to vertical and computes the precise grip pivot."""
    rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] != 4:
        return 0.5, 0.85

    # Only apply to hand_held (swords, wands, etc.)
    if anchor_type != "hand_held":
        return 0.5, 0.5 # default center for everything else

    # 1. PCA Rotation to make the prop perfectly vertical
    alpha = rgba[:, :, 3]
    moments = cv2.moments(alpha)
    if moments["m00"] > 0:
        mu20 = moments["mu20"]
        mu02 = moments["mu02"]
        mu11 = moments["mu11"]
        
        theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        angle_deg = math.degrees(theta)
        rotation_needed = 90 - angle_deg
        
        if abs(rotation_needed % 180) > 15 and abs(rotation_needed % 180) < 165:
            h, w = rgba.shape[:2]
            cx, cy = w // 2, h // 2
            M = cv2.getRotationMatrix2D((cx, cy), rotation_needed, 1.0)
            
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - cx
            M[1, 2] += (new_h / 2) - cy
            
            rgba = cv2.warpAffine(rgba, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            
            # Re-crop
            alpha = rgba[:, :, 3]
            coords = cv2.findNonZero(alpha)
            if coords is not None:
                x, y, w_b, h_b = cv2.boundingRect(coords)
                rgba = rgba[y:y+h_b, x:x+w_b]

    # 2. Ensure handle is at the bottom
    h, w = rgba.shape[:2]
    quarter = max(1, h // 4)
    alpha = rgba[:, :, 3]
    top_w = _avg_row_width(alpha[:quarter, :])
    bot_w = _avg_row_width(alpha[h - quarter:, :])
    
    if top_w > bot_w * 1.3:
        rgba = cv2.flip(rgba, 0)
    elif bot_w == 0 and top_w > 0:
        rgba = cv2.flip(rgba, 0)

    # 3. Dynamic Pivot Calculation (Find the crossguard)
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    
    handle_strip = alpha[int(h * 0.80):, :]
    col_mass = np.sum(handle_strip > 127, axis=0).astype(np.float64)
    total = np.sum(col_mass)
    pivot_x = float(np.sum(np.arange(w) * col_mass) / total) / w if total > 0 else 0.5
    
    bottom_half = alpha[h // 2:, :]
    widths = []
    for i, row in enumerate(bottom_half):
        cols = np.where(row > 127)[0]
        if len(cols) > 0:
            widths.append((i, cols[-1] - cols[0]))
        else:
            widths.append((i, 0))
            
    if widths:
        max_w_row, max_w = max(widths, key=lambda x: x[1])
        crossguard_y = (h // 2) + max_w_row
        if max_w > w * 0.15:
            pivot_y = min((crossguard_y + h * 0.05) / h, 0.95)
        else:
            pivot_y = 0.92
    else:
        pivot_y = 0.85

    # Overwrite the image with the perfectly upright, cropped version
    cv2.imwrite(image_path, rgba)
    
    return pivot_x, pivot_y
