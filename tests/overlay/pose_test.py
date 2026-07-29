import cv2
import math
import sys
from pathlib import Path

# Add project root to sys.path so backend.overlay can be imported
# when running this script directly (python tests/overlay/pose_test.py).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.overlay.pipeline import PoseAnchorPipeline, SKELETON_CONNECTIONS

# Change to test different anchors: "prop_in_hand", "right_wrist", "head",
# "both_shoulders", "both_wrists", "ambient"
REQUESTED_ANCHOR_TYPE = "left_wrist"

pipeline = PoseAnchorPipeline(anchor_type=REQUESTED_ANCHOR_TYPE)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print(f"Webcam opened. Testing anchor type: {REQUESTED_ANCHOR_TYPE}. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    payload, keypoints, resolved, _ = pipeline.process_frame(frame)
    h, w, _ = frame.shape

    if keypoints is not None:
        # SKELETON_CONNECTIONS uses landmark indices, but extract_keypoints only
        # keeps named ones — draw skeleton straight from named keypoints instead.
        name_by_idx = {
            0: "nose", 7: "left_ear", 8: "right_ear",
            11: "left_shoulder", 12: "right_shoulder",
            13: "left_elbow", 14: "right_elbow",
            15: "left_wrist", 16: "right_wrist",
        }
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            s_name, e_name = name_by_idx.get(start_idx), name_by_idx.get(end_idx)
            sp, ep = keypoints.get(s_name), keypoints.get(e_name)
            if sp and ep:
                sx, sy = int(sp["x"] * w), int(sp["y"] * h)
                ex, ey = int(ep["x"] * w), int(ep["y"] * h)
                cv2.line(frame, (sx, sy), (ex, ey), (255, 200, 0), 2)

    if payload is not None:
        points = payload["points"]

        # Draw a circle at EACH point (1 for single anchors, 2 for dual)
        scale_radius = int(30 * payload["scale"])
        for pt in points:
            cv2.circle(frame, (int(pt["x"]), int(pt["y"])), scale_radius, (0, 255, 0), 2)

        if len(points) == 2:
            # Dual anchor: draw a line connecting the two actual points
            # (e.g. the shoulder line itself, not a rotation vector off a midpoint)
            p1, p2 = points
            cv2.line(frame, (int(p1["x"]), int(p1["y"])), (int(p2["x"]), int(p2["y"])), (0, 0, 255), 3)
        else:
            # Single anchor: draw the rotation vector as before
            ax, ay = int(points[0]["x"]), int(points[0]["y"])
            angle_rad = math.radians(payload["angle"])
            line_len = scale_radius + 20
            lx = int(ax + line_len * math.sin(angle_rad))
            ly = int(ay - line_len * math.cos(angle_rad))
            cv2.line(frame, (ax, ay), (lx, ly), (0, 0, 255), 3)

        # On-screen telemetry
        label = f"ANCHOR: {payload['anchor_type']} | Scale: {payload['scale']}x | Angle: {payload['angle']}deg"
        cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        light_label = f"LIGHT -> Bri: {payload['brightness']} | Tint: {payload['tint']}"
        cv2.putText(frame, light_label, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        pts_label = f"POINTS: {len(points)} | FPS: {payload['fps']}"
        cv2.putText(frame, pts_label, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.putText(frame, f"FPS: {payload['fps'] if payload else 0:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("GhostFrame Spatial Overlay Engine Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
