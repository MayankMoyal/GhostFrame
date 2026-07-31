"""Prop alignment: turns a raw background-removed PNG into an
accurately-anchored asset.

Problem this solves
--------------------
`overlay.js` currently picks a grip point using a FIXED fraction of
the full canvas (ZONE_GRIP), e.g. "right_wrist": 0.85 meaning "85%
down the image". That assumption only holds if the object fills the
canvas edge-to-edge with no padding — but rembg's output is whatever
size Z-Image-Turbo generated, usually with a lot of empty transparent
margin around the actual object, and that margin varies every single
generation. A fixed fraction against a variable amount of padding is
why placement looks inconsistent/sloppy.

Fix, in two steps
------------------
1. Tight-crop the RGBA image to the real bounding box of its
   non-transparent pixels (`Image.getbbox(alpha_only=True)`). After
   this, "85% down the image" actually means 85% down the OBJECT.
2. For single-point "held" anchors (wrist/prop_in_hand), don't even
   trust a fixed fraction — estimate the object's own long axis via
   PCA on the alpha mask's pixel coordinates, and place the grip at
   the end of that axis nearest the bottom of the frame. This makes
   rotation/placement correct even when the model didn't draw the
   prop perfectly vertical (which it frequently won't).

Call `align_prop()` right after your existing `remove_background()`
call in main.py — see the integration note at the bottom of this file.
"""
import numpy as np
from PIL import Image

# Held/handheld props: use PCA long-axis detection instead of a fixed
# fraction, since these are the ones most likely to be drawn at an
# angle or with an irregular silhouette.
PCA_ANCHOR_TYPES = {"right_wrist", "left_wrist", "both_wrists"}

# Everything else keeps a fixed fraction, but now applied to the TIGHT
# bounding box instead of the full canvas — this alone is a big
# accuracy win with almost no extra compute.
BBOX_GRIP_FRACTION = {
    "head":            0.92,  # near the bottom edge of the object (the "wearing edge")
    "left_shoulder":   0.08,  # near the top edge (rests ON the shoulder)
    "right_shoulder":  0.08,
    "both_shoulders":  0.08,  # top edge; left/right corners used separately (see dual-point note)
    "ambient":         0.5,
    "background":      0.5,
}

ALPHA_THRESHOLD = 20  # pixels with alpha below this are treated as background noise


def tight_crop_to_content(rgba_image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop an RGBA image to the bounding box of its visible pixels.

    Returns (cropped_image, original_bbox). If the image is fully
    transparent (bad generation), returns the image unchanged with
    bbox=None — callers should treat that as a generation failure,
    not silently proceed with a meaningless pivot.
    """
    if rgba_image.mode != "RGBA":
        rgba_image = rgba_image.convert("RGBA")

    # getbbox(alpha_only=True) ignores RGB content and only looks at
    # the alpha channel — exactly the "real object extent" we want.
    bbox = rgba_image.getbbox(alpha_only=True)
    if bbox is None:
        return rgba_image, None

    return rgba_image.crop(bbox), bbox


def _alpha_mask_coords(rgba_image: Image.Image) -> np.ndarray:
    """Pixel coordinates (row, col) of all non-transparent pixels."""
    alpha = np.array(rgba_image.split()[-1])
    ys, xs = np.where(alpha > ALPHA_THRESHOLD)
    return np.stack([xs, ys], axis=1)  # shape (N, 2), columns = (x, y)


def _pca_long_axis(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal axis of a point cloud via covariance eigen-decomposition.

    Returns (centroid, unit_direction_vector) for the axis of greatest
    spread — for an elongated object like a sword, this runs along its
    length regardless of what angle it was drawn at.
    """
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; the LAST eigenvector is the
    # direction of greatest variance = the long axis.
    long_axis = eigvecs[:, -1]
    return centroid, long_axis


def compute_grip_pca(cropped_rgba: Image.Image) -> dict:
    """Grip point + intrinsic angle for elongated held props, via PCA.

    Returns
    -------
    dict with:
        grip_x, grip_y : float in [0, 1], normalized within the
            cropped image — the end of the object's long axis closest
            to the bottom of the frame (the "handle" end, on the
            reasonable assumption a held prop is generated roughly
            upright with the grip lower than the tip).
        intrinsic_angle_deg : float — how far the object's long axis
            deviates from vertical in the image AS DRAWN. The overlay
            should subtract this from the live tracked angle so a
            prop drawn slightly tilted still ends up visually
            straight relative to the tracked limb.
    """
    w, h = cropped_rgba.size
    coords = _alpha_mask_coords(cropped_rgba)

    if len(coords) < 10:
        # Not enough signal (near-empty mask) — fall back to bottom-center.
        return {"grip_x": 0.5, "grip_y": 0.9, "intrinsic_angle_deg": 0.0}

    centroid, axis = _pca_long_axis(coords)

    # Project all points onto the axis to find its two extreme ends.
    projections = (coords - centroid) @ axis
    end_a = centroid + axis * projections.min()
    end_b = centroid + axis * projections.max()

    # The "grip" end is whichever endpoint has the larger y (lower in
    # the image, since image y grows downward) — the composition_hint
    # asked the model for handle-at-bottom, so this picks that end
    # even if the model didn't draw it perfectly vertical.
    grip_point = end_a if end_a[1] > end_b[1] else end_b
    tip_point = end_b if end_a[1] > end_b[1] else end_a

    grip_x = float(np.clip(grip_point[0] / w, 0.0, 1.0))
    grip_y = float(np.clip(grip_point[1] / h, 0.0, 1.0))

    # Angle of the grip->tip vector relative to "straight up" (0,-1),
    # matching the backend's atan2(dx, -dy) convention so it composes
    # directly with the live tracked angle.
    dx = tip_point[0] - grip_point[0]
    dy = tip_point[1] - grip_point[1]
    intrinsic_angle = np.degrees(np.arctan2(dx, -dy))

    return {
        "grip_x": round(grip_x, 4),
        "grip_y": round(grip_y, 4),
        "intrinsic_angle_deg": round(float(intrinsic_angle), 1),
    }


def compute_grip_bbox_fraction(anchor_type: str) -> dict:
    """Fixed-fraction grip for non-elongated anchors, applied to the
    TIGHT bbox (not the full canvas) — accurate because padding is
    already gone by the time this runs."""
    fraction = BBOX_GRIP_FRACTION.get(anchor_type, 0.5)
    return {"grip_x": 0.5, "grip_y": fraction, "intrinsic_angle_deg": 0.0}


def align_prop(rgba_image: Image.Image, anchor_type: str) -> dict:
    """Full alignment step: tight-crop + compute an accurate grip point.

    Call this AFTER remove_background() produces an RGBA image, BEFORE
    saving the final file that gets served to the overlay.

    Returns
    -------
    dict with:
        image        : the tight-cropped PIL Image (SAVE THIS, not the
                        original — its dimensions are what grip_x/
                        grip_y are normalized against)
        grip_x, grip_y, intrinsic_angle_deg : see above
        crop_bbox    : the box that was cropped from the original, for
                        debugging/logging
    """
    cropped, bbox = tight_crop_to_content(rgba_image)

    if bbox is None:
        # Fully transparent generation — nothing usable came out of
        # the model. Caller should treat this as a failed generation,
        # not proceed to broadcast a blank prop.
        return {
            "image": rgba_image, "grip_x": 0.5, "grip_y": 0.5,
            "intrinsic_angle_deg": 0.0, "crop_bbox": None, "valid": False,
        }

    if anchor_type in PCA_ANCHOR_TYPES:
        grip = compute_grip_pca(cropped)
    else:
        grip = compute_grip_bbox_fraction(anchor_type)

    return {
        "image": cropped,
        "grip_x": grip["grip_x"],
        "grip_y": grip["grip_y"],
        "intrinsic_angle_deg": grip["intrinsic_angle_deg"],
        "crop_bbox": bbox,
        "valid": True,
    }


# ---------------------------------------------------------------------
# Integration note for main.py:
#
#   from postprocess.prop_alignment import align_prop
#
#   ... after remove_background(output_path, nobg_path) succeeds ...
#   rgba = Image.open(nobg_path).convert("RGBA")
#   result = align_prop(rgba, agent_result.get("anchor_type", "background"))
#   if not result["valid"]:
#       raise HTTPException(status_code=500, detail="Generation produced no visible content.")
#   result["image"].save(nobg_path)  # overwrite with the tight-cropped version
#
#   broadcast_msg = {
#       "type": "new_prop",
#       "filename": nobg_filename,
#       "anchor_type": agent_result.get("anchor_type", "background"),
#       "grip_x": result["grip_x"],
#       "grip_y": result["grip_y"],
#       "intrinsic_angle_deg": result["intrinsic_angle_deg"],
#   }
# ---------------------------------------------------------------------
