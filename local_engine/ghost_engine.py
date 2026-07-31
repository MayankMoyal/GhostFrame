"""Ghost Engine -- The master pipeline for Ghost Frame.

This script integrates:
1. Real-time webcam feed
2. Robust Video Matting (RVM) to isolate the user
3. AI-generated background replacement
4. Universal Prop Engine with Depth-Aware Compositing (Z-Index)
5. Virtual Camera output to OBS
6. Local Bridge API for Web UI integration
"""
import argparse
import time
import os
import glob
import cv2
import numpy as np
import pyvirtualcam
import threading
import requests
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from tracker import UltimateTracker
from background_remover import BackgroundRemover
from prop_manager import PropManager
from prop_config import load_pipeline_json, resolve_category
from live_equip import render_prop, _ScaleFilter, _PropState

# --- LOCAL BRIDGE API ---
bridge_app = FastAPI()
connected_websockets = set()
latest_tracking_payload = None

# Globals for passing data between threads
active_props_dir = "active_props"
force_reload_props = False

@bridge_app.post("/equip")
async def equip_prop(data: dict):
    """Called by the Web UI when a new prop is generated on the Cloud."""
    global force_reload_props
    url = data.get("url")
    if url:
        print(f"\n[Bridge API] ⬇️ Downloading new prop from Cloud: {url}")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            
            # Clear old props to prevent stacking
            for old_file in os.listdir(active_props_dir):
                os.remove(os.path.join(active_props_dir, old_file))
                
            filename = os.path.basename(url)
            save_path = os.path.join(active_props_dir, filename)
            with open(save_path, "wb") as f:
                f.write(resp.content)
            print(f"[Bridge API] ✅ Saved to {save_path}. Hot-swapping now!")
            force_reload_props = True
        except Exception as e:
            print(f"[Bridge API] ❌ Failed to download prop: {e}")
            
    return {"status": "ok"}

@bridge_app.websocket("/ws/anchor")
async def ws_anchor(websocket: WebSocket):
    """Streams live tracking coordinates to overlay.html in OBS."""
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            if latest_tracking_payload:
                await websocket.send_json(latest_tracking_payload)
            await asyncio.sleep(1/30)  # 30 fps
    except WebSocketDisconnect:
        connected_websockets.discard(websocket)

def run_bridge_api():
    import logging
    log = logging.getLogger("uvicorn")
    log.setLevel(logging.WARNING)  # keep terminal clean
    uvicorn.run(bridge_app, host="127.0.0.1", port=8001, log_level="warning")


# --- ENGINE CORE ---

def load_props_from_folder(folder_path, hand):
    """Loads all PNG/JPG images in the folder as props."""
    active_props = []
    if not os.path.exists(folder_path):
        return active_props
        
    for file in glob.glob(os.path.join(folder_path, "*.*")):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        cat = None
        for cat_enum in ["hand_held", "shield", "head_wear", "neck_wear", "wrist_wear", "ear_wear", "face_wear", "body_wear"]:
            if cat_enum in file.lower():
                cat = resolve_category(cat_enum)
                break
                
        meta = PropManager.process_image(file, category=cat, target_side=hand)
        if meta:
            active_props.append({"meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()})
            
    return active_props

def main():
    global active_props_dir
    global force_reload_props
    global latest_tracking_payload
    
    ap = argparse.ArgumentParser(description="Ghost Engine - Live Pipeline")
    ap.add_argument("--props-dir", default="active_props", help="Directory to watch for active props")
    ap.add_argument("--bg", dest="background", default=None, help="AI generated background image path")
    ap.add_argument("--hand", choices=["left", "right", "any"], default="any", help="Default hand/side for props")
    ap.add_argument("--debug", action="store_true", help="Show landmark dots")
    ap.add_argument("--no-mirror", action="store_true", help="Disable selfie mirror")
    args = ap.parse_args()

    active_props_dir = args.props_dir
    os.makedirs(active_props_dir, exist_ok=True)
    
    # 1. Start Bridge API Background Thread
    threading.Thread(target=run_bridge_api, daemon=True).start()
    print("[Ghost Engine] Local Bridge API running on http://127.0.0.1:8001")
    
    # 2. Load Background
    if args.background:
        bg_image = cv2.imread(args.background)
    else:
        bg_image = np.zeros((480, 640, 3), dtype=np.uint8)
        bg_image[:] = (40, 40, 40)

    # 3. Initialize Engines
    tracker = UltimateTracker()
    remover = BackgroundRemover()

    # 4. Camera Loop
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    ok, raw = cap.read()
    if not ok:
        print("[ERROR] Cannot read from webcam.")
        return
    h, w = raw.shape[:2]
    mirror = not args.no_mirror

    active_props = []
    back_props = []
    front_props = []

    print(f"\n[Ghost Engine] SYSTEM READY. Broadcasting Virtual Camera ({w}x{h}). Press Ctrl+C in terminal to quit.\n")

    with pyvirtualcam.Camera(width=w, height=h, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        while True:
            # Hot-swap check
            if force_reload_props:
                active_props = load_props_from_folder(active_props_dir, args.hand)
                back_props = [p for p in active_props if p["meta"].profile.z_index < 0]
                front_props = [p for p in active_props if p["meta"].profile.z_index >= 0]
                force_reload_props = False

            ok, raw = cap.read()
            if not ok:
                break
                
            display_frame = cv2.flip(raw, 1) if mirror else raw.copy()

            # Track user
            track_result = tracker.detect(raw)
            
            # --- WebSocket Payload Construction for overlay.html ---
            # We construct a simple tracking payload based on the right wrist (or shoulder if no wrist)
            # This allows overlay.html to move the images around.
            ws_x, ws_y = 0.5, 0.5
            ws_scale = 1.0
            if track_result.pose_landmarks:
                import mediapipe as mp
                from mediapipe.tasks.vision.pose_landmarker import PoseLandmark
                landmarks = track_result.pose_landmarks
                # Try right wrist (index 16). In MediaPipe, x and y are normalized [0.0, 1.0].
                rw = landmarks[16]
                if rw.visibility > 0.5:
                    ws_x, ws_y = rw.x, rw.y
                else:
                    # Fallback to shoulder
                    rs = landmarks[12]
                    ws_x, ws_y = rs.x, rs.y
                    
                # Mirror coordinate for the overlay if needed
                if mirror:
                    ws_x = 1.0 - ws_x
                    
            latest_tracking_payload = {
                "points": [{"x": ws_x * w, "y": ws_y * h}],
                "angle": 0,  # Simplify angle for HTML overlay
                "scale": ws_scale,
                "frame_width": w,
                "frame_height": h
            }
            # -----------------------------------------------------

            # Render Background Props -> RVM -> Foreground Props (Virtual Camera Pipeline)
            current_bg = cv2.resize(bg_image, (w, h))
            for pdata in back_props:
                current_bg = render_prop(current_bg, pdata, track_result, mirror, w, h, args.debug)

            comp = remover.remove(display_frame, current_bg)

            for pdata in front_props:
                comp = render_prop(comp, pdata, track_result, mirror, w, h, args.debug)

            # Broadcast to OBS Virtual Camera
            cam.send(comp)
            cam.sleep_until_next_frame()
            
            cv2.imshow("Ghost Frame (Virtual Cam)", comp)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    remover.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
