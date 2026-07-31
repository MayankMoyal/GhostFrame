"""Prop Manager -- Category-aware background removal, orientation, pivot, and metadata.

The processing pipeline adapts to the prop category:
- HAND_HELD  -> auto-orient vertical (blade up, handle down), detect handle pivot
- HEAD_WEAR  -> keep as-is, pivot at bottom centre (sits on head)
- NECK_WEAR  -> keep as-is, pivot at top centre (hangs from neck)
- FACE_WEAR  -> keep as-is, pivot at centre (bridge of glasses / mask centre)
- EAR_WEAR   -> keep as-is, pivot at top centre (earring hook)
- SHIELD     -> keep as-is, pivot at centre (held on forearm)
- BODY_WEAR  -> keep as-is, pivot at top centre (drapes from shoulders)
- WRIST_WEAR -> keep as-is, pivot at centre (wraps around wrist)
"""
import io
import os
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rembg
from PIL import Image

from prop_config import (
    AttachmentProfile,
    PropCategory,
    PROFILES,
    classify_from_prompt,
    resolve_category,
)


@dataclass
class PropMetadata:
    """Everything the renderer needs to composite one prop."""
    image: np.ndarray                       # RGBA, processed and oriented
    image_flipped: np.ndarray               # Horizontally-flipped copy
    pivot: Tuple[float, float]              # Normalised pivot (0-1) for default hand/side
    pivot_flipped: Tuple[float, float]      # Pivot for the flipped version
    handle_width: int                       # Handle width in px (meaningful for HAND_HELD)
    category: PropCategory
    profile: AttachmentProfile
    aspect_ratio: float                     # width / height
    target_side: str                        # "left" | "right" | "any"


class PropManager:
    """Static utility class for loading, processing, and configuring prop assets."""

    # --- Public API ---
    @staticmethod
    def process_image(
        image_path: str,
        category: Optional[PropCategory] = None,
        prompt: str = "",
        target_side: str = "any",
    ) -> Optional[PropMetadata]:
        """Full pipeline: load -> BG remove -> crop -> orient -> pivot -> metadata.

        Parameters
        ----------
        image_path : str
            Path to the prop image (PNG / JPG / WEBP).
        category : PropCategory, optional
            Explicit category.  Falls back to keyword classification on *prompt*,
            then to HAND_HELD.
        prompt : str
            Original user prompt (used for fallback classification).
        target_side : str
            "left" | "right" | "any" for paired body parts.
        """
        # Resolve category
        if category is None:
            if prompt:
                category = classify_from_prompt(prompt)
            else:
                category = PropCategory.HAND_HELD
        profile = PROFILES[category]

        print(f"[PropManager] Processing '{image_path}' as {category.value} ...")

        try:
            with open(image_path, "rb") as f:
                input_data = f.read()
            output_data = rembg.remove(input_data)
            img = Image.open(io.BytesIO(output_data)).convert("RGBA")
            arr = np.array(img)
            bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            alpha = arr[:, :, 3]
            rgba = np.dstack((bgr, alpha))

            # Tight crop
            coords = cv2.findNonZero(alpha)
            if coords is None:
                print("[PropManager] Error: no visible object after BG removal.")
                return None
            x, y, w, h = cv2.boundingRect(coords)
            cropped = rgba[y:y + h, x:x + w]
            print(f"[PropManager] Cropped to {w}x{h}.")

            # Category-specific orientation
            cropped = PropManager._orient(cropped, profile)

            # Category-specific pivot detection
            pivot = PropManager._detect_pivot(cropped, profile)

            # Handle width (only meaningful for HAND_HELD, but always computed)
            handle_width = PropManager._get_handle_width(cropped)

            # Pre-compute flipped version
            image_flipped = cv2.flip(cropped, 1)
            pivot_flipped = (1.0 - pivot[0], pivot[1])

            aspect_ratio = cropped.shape[1] / max(cropped.shape[0], 1)

            meta = PropMetadata(
                image=cropped,
                image_flipped=image_flipped,
                pivot=pivot,
                pivot_flipped=pivot_flipped,
                handle_width=handle_width,
                category=category,
                profile=profile,
                aspect_ratio=aspect_ratio,
                target_side=target_side,
            )
            print(
                f"[PropManager] Done. {cropped.shape[1]}x{cropped.shape[0]}, "
                f"Pivot=({pivot[0]:.2f},{pivot[1]:.2f}), "
                f"Handle={handle_width}px, Cat={category.value}"
            )
            return meta

        except Exception as e:
            print(f"[PropManager] Error: {e}")
            traceback.print_exc()
            return None

    # --- Orientation ---
    @staticmethod
    def _orient(rgba: np.ndarray, profile: AttachmentProfile) -> np.ndarray:
        mode = profile.auto_orient
        if mode == "vertical_handle_down":
            return PropManager._orient_vertical_handle_down(rgba)
        # "as_is" -- no changes
        return rgba

    @staticmethod
    def _orient_vertical_handle_down(rgba: np.ndarray) -> np.ndarray:
        """Rotate horizontal props to vertical, ensure handle is at bottom."""
        h, w = rgba.shape[:2]
        alpha = rgba[:, :, 3]

        if w > h * 1.3:
            rgba = cv2.rotate(rgba, cv2.ROTATE_90_CLOCKWISE)
            h, w = rgba.shape[:2]
            alpha = rgba[:, :, 3]
            print("[PropManager] Auto-rotated horizontal -> vertical.")

        quarter = max(1, h // 4)
        top_w = PropManager._avg_row_width(alpha[:quarter, :])
        bot_w = PropManager._avg_row_width(alpha[h - quarter:, :])
        if top_w > bot_w * 1.3:
            rgba = cv2.flip(rgba, 0)
            print("[PropManager] Flipped so handle is at bottom.")
        return rgba

    @staticmethod
    def _avg_row_width(strip: np.ndarray, threshold: int = 127) -> float:
        widths = []
        for row in strip:
            cols = np.where(row > threshold)[0]
            if len(cols) > 0:
                widths.append(cols[-1] - cols[0])
        return float(np.mean(widths)) if widths else 0.0

    # --- Pivot Detection ---
    @staticmethod
    def _detect_pivot(
        rgba: np.ndarray, profile: AttachmentProfile
    ) -> Tuple[float, float]:
        strategy = profile.pivot_strategy

        if strategy == "handle_bottom":
            return PropManager._pivot_handle_bottom(rgba)
        elif strategy == "bottom_center":
            return PropManager._pivot_bottom_center(rgba)
        elif strategy == "top_center":
            return PropManager._pivot_top_center(rgba)
        elif strategy == "center":
            return PropManager._pivot_center(rgba)

        return profile.pivot_default

    @staticmethod
    def _pivot_handle_bottom(rgba: np.ndarray) -> Tuple[float, float]:
        """Alpha-mask centroid of the bottom 20% for hand-held props."""
        h, w = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        handle_strip = alpha[int(h * 0.80):, :]
        col_mass = np.sum(handle_strip > 127, axis=0).astype(np.float64)
        total = np.sum(col_mass)
        if total > 0:
            pivot_x = float(np.sum(np.arange(w) * col_mass) / total) / w
        else:
            pivot_x = 0.5
        return (pivot_x, 0.85)

    @staticmethod
    def _pivot_bottom_center(rgba: np.ndarray) -> Tuple[float, float]:
        """Bottom-centre, e.g. where a hat sits on the head."""
        h, w = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        # Find horizontal centroid of the bottom 10%
        strip = alpha[int(h * 0.90):, :]
        col_mass = np.sum(strip > 127, axis=0).astype(np.float64)
        total = np.sum(col_mass)
        if total > 0:
            cx = float(np.sum(np.arange(w) * col_mass) / total) / w
        else:
            cx = 0.5
        return (cx, 0.95)

    @staticmethod
    def _pivot_top_center(rgba: np.ndarray) -> Tuple[float, float]:
        """Top-centre, e.g. where a necklace chain meets the neck."""
        h, w = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        strip = alpha[:int(h * 0.10) + 1, :]
        col_mass = np.sum(strip > 127, axis=0).astype(np.float64)
        total = np.sum(col_mass)
        if total > 0:
            cx = float(np.sum(np.arange(w) * col_mass) / total) / w
        else:
            cx = 0.5
        return (cx, 0.05)

    @staticmethod
    def _pivot_center(rgba: np.ndarray) -> Tuple[float, float]:
        """Centre of mass of the whole alpha mask."""
        alpha = rgba[:, :, 3]
        moments = cv2.moments(alpha, binaryImage=False)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"] / rgba.shape[1]
            cy = moments["m01"] / moments["m00"] / rgba.shape[0]
            return (cx, cy)
        return (0.5, 0.5)

    # --- Handle Width ---
    @staticmethod
    def _get_handle_width(
        rgba: np.ndarray, alpha_threshold: int = 127
    ) -> int:
        """Median opaque-pixel span across the bottom 15%."""
        h, w = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        strip = alpha[int(h * 0.85):, :]
        binary = strip > alpha_threshold
        row_widths = []
        for row in binary:
            cols = np.where(row)[0]
            if len(cols) > 0:
                row_widths.append(cols[-1] - cols[0])
        return int(np.median(row_widths)) if row_widths else w
