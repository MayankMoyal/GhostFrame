import cv2
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import math

# Add project root to sys.path so we can import the backend code
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.overlay.pipeline import PoseAnchorPipeline

# 1. Read the image path from the command line
if len(sys.argv) < 2:
    print("Usage: python tests/overlay/prop_test.py <path_to_transparent_image.png> [anchor_type]")
    sys.exit(1)

img_path = sys.argv[1]
try:
    # Load the sword image using Pillow (which handles transparency perfectly)
    prop_img = Image.open(img_path).convert("RGBA")
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

# 2. Read anchor type from command line (optional, defaults to right_wrist)
#    Usage: python prop_test.py sword.png right_wrist
ANCHOR_TYPE = sys.argv[2] if len(sys.argv) > 2 else "right_wrist"

# How big should the prop's longest side be, relative to shoulder width?
# This is tuned per anchor ZONE so any object placed there looks natural.
ZONE_SCALE = {
    "right_wrist":     2.0,   # wrist props ≈ arm length ≈ 2x shoulder width
    "left_wrist":      2.0,
    "prop_in_hand":    2.0,
    "both_wrists":     2.5,
    "head":            1.0,   # head props (hats, crowns) ≈ shoulder width
    "left_shoulder":   1.5,
    "right_shoulder":  1.5,
    "both_shoulders":  2.5,   # shoulder-span props (capes, wings)
    "ambient":         1.5,
    "background":      1.5,
}

# Where is the "grip" on the image? (0.0 = top, 0.5 = center, 1.0 = bottom)
# For a sword, the handle is at the bottom → 0.85 (slightly above absolute bottom)
# For a hat, the attachment is at the bottom → 0.9
# For wings, the attachment is at the center → 0.5
ZONE_GRIP = {
    "right_wrist":     0.85,
    "left_wrist":      0.85,
    "prop_in_hand":    0.85,
    "both_wrists":     0.5,
    "head":            0.9,
    "left_shoulder":   0.5,
    "right_shoulder":  0.5,
    "both_shoulders":  0.5,
    "ambient":         0.5,
    "background":      0.5,
}

pipeline = PoseAnchorPipeline(anchor_type=ANCHOR_TYPE)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

zone_ratio = ZONE_SCALE.get(ANCHOR_TYPE, 1.0)
grip_ratio = ZONE_GRIP.get(ANCHOR_TYPE, 0.5)
print(f"Webcam opened. Anchor: {ANCHOR_TYPE} | Zone scale: {zone_ratio}x | Grip: {grip_ratio}")
print(f"Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    payload, keypoints, resolved, mask = pipeline.process_frame(frame)
    
    if payload is not None and payload["points"] and keypoints is not None:
        pt = payload["points"][0]
        h, w, _ = frame.shape
        
        # === BACKGROUND COMPOSITING ===
        if ANCHOR_TYPE == "background":
            # Resize the prop (which acts as the background) to fit the whole frame
            bg_resized = cv2.resize(np.array(prop_img.convert("RGB"))[:, :, ::-1], (w, h))
            
            if mask is not None:
                # The mask from MediaPipe is [0.0, 1.0] where 1.0 is the person.
                # Resize mask to frame size.
                mask_resized = cv2.resize(mask, (w, h))
                
                # Apply a Gaussian blur to the mask to feather the edges and remove jitter
                mask_blurred = cv2.GaussianBlur(mask_resized, (21, 21), 0)
                
                # Convert mask to 3 channels for blending
                mask_3d = np.expand_dims(mask_blurred, axis=-1)
                
                # Blend: (Person * mask) + (Background * (1 - mask))
                frame = (frame * mask_3d + bg_resized * (1.0 - mask_3d)).astype(np.uint8)
            else:
                # Fallback: If your MediaPipe model version doesn't support masks,
                # just blend the background at 50% opacity so you can still see yourself.
                # (In the real OBS frontend, this is handled by a proper green screen filter).
                frame = cv2.addWeighted(frame, 0.5, bg_resized, 0.5, 0)
                
        # === FOREGROUND PROP PASTING ===
        else:
            # === UNIVERSAL BODY-RELATIVE SCALING ===
            #
            # We use SHOULDER WIDTH (in pixels) as the universal body reference.
            # It's the most stable landmark — always visible, scales perfectly
            # with distance, and gives a natural "human sized" reference.
            #
            # The prop's longest dimension is set to:
            #   shoulder_width_px × zone_ratio
            #
            # This works for ANY object because we're not measuring the object,
            # we're measuring the PERSON. A sword at the wrist, a crown on the
            # head, wings on the shoulders — all scale naturally.
            
            shoulder_px = None
            ls = keypoints.get("left_shoulder")
            rs = keypoints.get("right_shoulder")
            if (ls and rs 
                    and ls.get("visibility", 0) > 0.5 
                    and rs.get("visibility", 0) > 0.5):
                shoulder_px = (((ls["x"] - rs["x"]) * w) ** 2 + ((ls["y"] - rs["y"]) * h) ** 2) ** 0.5
            
            if shoulder_px is None or shoulder_px < 20:
                # Fallback: use ear-to-ear distance × 3 (ears ≈ 1/3 shoulder width)
                le = keypoints.get("left_ear")
                re = keypoints.get("right_ear")
                if (le and re
                        and le.get("visibility", 0) > 0.5
                        and re.get("visibility", 0) > 0.5):
                    ear_px = (((le["x"] - re["x"]) * w) ** 2 + ((le["y"] - re["y"]) * h) ** 2) ** 0.5
                    shoulder_px = ear_px * 3.0
                else:
                    shoulder_px = w * 0.25  # last resort: 25% of frame width
            
            target_size = shoulder_px * zone_ratio
            scale_factor = target_size / max(prop_img.width, prop_img.height)
            angle = payload["angle"]
            
            new_w = int(prop_img.width * scale_factor)
            new_h = int(prop_img.height * scale_factor)
            
            if new_w > 10 and new_h > 10:
                prop_resized = prop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Angle convention from pipeline: 0° = arm pointing UP, positive = CW.
                # Pillow's rotate() is counter-clockwise, so negate to align.
                pil_angle = -angle
                prop_rotated = prop_resized.rotate(pil_angle, expand=True)
                
                # === GRIP POINT MATH ===
                # The "grip" is where the prop attaches to the body (e.g. sword handle).
                # grip_ratio=0.85 means 85% down the image (near the bottom = handle).
                # We calculate where that grip point ends up AFTER rotation,
                # then offset the paste so that exact point lands on the wrist.
                
                # Grip point relative to the CENTER of the pre-rotation image:
                #   x offset = 0 (horizontally centered)
                #   y offset = (grip_ratio - 0.5) * new_h  (positive = below center)
                grip_dy = (grip_ratio - 0.5) * new_h
                
                # After Pillow's CCW rotation by pil_angle degrees, the offset rotates:
                pil_rad = math.radians(pil_angle)
                rotated_grip_x = grip_dy * math.sin(pil_rad)
                rotated_grip_y = grip_dy * math.cos(pil_rad)
                
                # Grip point in the rotated image's pixel coordinates:
                grip_in_img_x = prop_rotated.width / 2 + rotated_grip_x
                grip_in_img_y = prop_rotated.height / 2 + rotated_grip_y
                
                # Paste so the grip point is exactly on the wrist
                paste_x = int(pt["x"] - grip_in_img_x)
                paste_y = int(pt["y"] - grip_in_img_y)
                
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                pil_frame = Image.fromarray(frame_rgba)
                
                pil_frame.paste(prop_rotated, (paste_x, paste_y), prop_rotated)
                frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGBA2BGR)
            
    cv2.imshow("GhostFrame Local Prop Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

