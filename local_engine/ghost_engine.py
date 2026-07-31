"""Ghost Engine — The master pipeline for Ghost Stream (Local PC).

Integrates:
1. Real-time webcam feed (30fps)
2. MediaPipe pose/hand tracking (tracker.py)
3. Robust Video Matting (background_remover.py) to isolate the user
4. AI-generated background replacement
5. Universal Prop Engine with Depth-Aware Compositing (live_equip.py)
6. Local API (Port 8001) for receiving props from the Cloud Backend
7. WebSocket server broadcasting tracking data to OBS overlay
8. Heartbeat to Cloud Backend for health dashboard

Pipeline per frame:
1. Run MediaPipe tracker on the raw frame to get pose/hand landmarks.
2. Resize AI background to match frame dimensions.
3. Render 'Behind' props (Z < 0) onto the AI background.
4. Run RVM to blend the user's foreground over the AI background.
5. Render 'Front' props (Z >= 0) onto the final composited frame.
6. Broadcast tracking data via WebSocket to overlay clients.
7. Display composite frame in OpenCV window (for OBS Window Capture).
"""
import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracker import UltimateTracker
from background_remover import BackgroundRemover
from prop_manager import PropManager
from prop_config import load_pipeline_json, resolve_category
from live_equip import render_prop, _ScaleFilter, _PropState

try:
    from config import (
        LOCAL_HOST, LOCAL_PORT, CLOUD_API_URL,
        CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, MIRROR_FEED,
        HEARTBEAT_INTERVAL_SECONDS,
    )
except ImportError:
    LOCAL_HOST = "0.0.0.0"
    LOCAL_PORT = 8001
    CLOUD_API_URL = "http://localhost:8000"
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    MIRROR_FEED = True
    HEARTBEAT_INTERVAL_SECONDS = 5


# ── Shared State ──────────────────────────────────────────────────────────
# These are accessed from both the camera thread and the API thread.

# Current tracking data (updated every frame by camera thread)
latest_tracking_data = {}
tracking_lock = threading.Lock()

# Active props (can be modified by API thread)
active_props = []
props_lock = threading.Lock()

# Background image (can be updated by API)
bg_image_holder = {"image": None}
bg_lock = threading.Lock()

# WebSocket overlay clients
ws_clients = set()
ws_clients_lock = threading.Lock()

# Engine status for heartbeat
engine_status = {
    "running": False,
    "fps": 0.0,
    "prop_count": 0,
    "camera_resolution": f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
}


# ── WebSocket Broadcast ──────────────────────────────────────────────────
async def _ws_broadcast(message: str):
    """Broadcast a message to all connected WebSocket overlay clients."""
    dead = set()
    with ws_clients_lock:
        clients = set(ws_clients)
    for ws in clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    if dead:
        with ws_clients_lock:
            ws_clients.difference_update(dead)


def broadcast_tracking_sync(data: dict):
    """Non-async version for calling from the camera thread."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                _ws_broadcast(json.dumps(data)), loop
            )
    except RuntimeError:
        pass


# ── Camera Loop (runs in its own thread) ─────────────────────────────────
def camera_loop(args):
    """Main camera capture and processing loop."""
    global engine_status

    # 1. Load initial props from CLI
    with props_lock:
        if args.pipeline_json:
            requests = load_pipeline_json(args.pipeline_json)
            for req in requests:
                cat = resolve_category(req.category)
                side = req.target_side or args.hand
                meta = PropManager.process_image(
                    req.image_path, category=cat, prompt=req.prompt,
                    target_side=side)
                if meta:
                    active_props.append({
                        "meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()
                    })

        for spec in args.props:
            if ":" in spec:
                path, cat_str = spec.rsplit(":", 1)
                cat = resolve_category(cat_str)
            else:
                path = spec
                cat = None
            meta = PropManager.process_image(path, category=cat, target_side=args.hand)
            if meta:
                active_props.append({
                    "meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()
                })

    # 2. Load Background
    with bg_lock:
        if args.background:
            bg_image_holder["image"] = cv2.imread(args.background)
            if bg_image_holder["image"] is None:
                print(f"[ERROR] Could not read background: '{args.background}'")
                bg_image_holder["image"] = np.full((480, 640, 3), 40, dtype=np.uint8)
            else:
                print(f"[Ghost Engine] Using AI background: {args.background}")
        else:
            bg_image_holder["image"] = np.full((480, 640, 3), 40, dtype=np.uint8)
            print("[Ghost Engine] No background provided. Using default dark gray.")

    # 3. Initialize Engines
    print("[Ghost Engine] Initializing MediaPipe Tracker...")
    tracker = UltimateTracker()

    print("[Ghost Engine] Initializing RVM Background Remover...")
    remover = BackgroundRemover()

    # 4. Camera Capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    mirror = MIRROR_FEED and not args.no_mirror
    print(f"\n[Ghost Engine] SYSTEM READY. Camera: {CAMERA_INDEX}, "
          f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}, Mirror: {mirror}")
    print("[Ghost Engine] Press ESC in the video window to quit.\n")

    engine_status["running"] = True
    fps_t, fps_n, fps_val = time.time(), 0, 0.0

    while True:
        ok, raw = cap.read()
        if not ok:
            break

        h, w = raw.shape[:2]
        display_frame = cv2.flip(raw, 1) if mirror else raw.copy()

        # Step A: Track user on the raw unmirrored frame
        track_result = tracker.detect(raw)

        # Step B: Prepare Background
        with bg_lock:
            current_bg = cv2.resize(bg_image_holder["image"], (w, h))

        # Step C: Sort props by Z-index and render
        with props_lock:
            back_props = [p for p in active_props if p["meta"].profile.z_index < 0]
            front_props = [p for p in active_props if p["meta"].profile.z_index >= 0]

        # Step C1: Render 'Behind' props onto the background
        for pdata in back_props:
            current_bg = render_prop(current_bg, pdata, track_result, mirror, w, h, args.debug)

        # Step D: Background Removal (Composites User over current_bg)
        comp = remover.remove(display_frame, current_bg)

        # Step E: Render 'Front' Props onto the composited image
        for pdata in front_props:
            comp = render_prop(comp, pdata, track_result, mirror, w, h, args.debug)

        # Step F: Build and broadcast tracking data for OBS overlay
        tracking_payload = _build_tracking_payload(track_result, w, h)
        with tracking_lock:
            latest_tracking_data.update(tracking_payload)

        # Broadcast to WebSocket clients
        broadcast_tracking_sync(tracking_payload)

        # FPS Counter
        fps_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            fps_val = fps_n / elapsed
            fps_n = 0
            fps_t = time.time()
            engine_status["fps"] = round(fps_val, 1)
            engine_status["prop_count"] = len(active_props)

        cv2.putText(comp, f"FPS: {fps_val:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Ghost Stream", comp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    engine_status["running"] = False
    cap.release()
    remover.close()
    cv2.destroyAllWindows()


def _build_tracking_payload(track_result, frame_w, frame_h) -> dict:
    """Convert tracker results to the format expected by overlay.js."""
    payload = {
        "frame_width": frame_w,
        "frame_height": frame_h,
        "angle": 0,
        "scale": 1.0,
        "brightness": 1.0,
        "points": [],
    }

    if track_result is None:
        payload["error"] = "no_detection"
        return payload

    # Use pose data for tracking points
    if track_result.pose:
        pose = track_result.pose

        # Calculate scale from shoulder width
        shoulder_width = getattr(pose, 'shoulder_width', 0)
        if shoulder_width > 0:
            payload["scale"] = shoulder_width / (0.25 * frame_w)

        # Add key tracking points
        points = []

        # Right wrist
        if track_result.hands:
            for hand in track_result.hands:
                points.append({
                    "x": hand.palm_x * frame_w,
                    "y": hand.palm_y * frame_h,
                    "side": hand.side,
                })

        # Head
        head_center = getattr(pose, 'head_center', None)
        if head_center:
            points.append({
                "x": head_center[0] * frame_w,
                "y": head_center[1] * frame_h,
                "zone": "head",
            })

        # Shoulders midpoint
        neck_point = getattr(pose, 'neck_point', None)
        if neck_point:
            points.append({
                "x": neck_point[0] * frame_w,
                "y": neck_point[1] * frame_h,
                "zone": "both_shoulders",
            })

        payload["points"] = points

        # Head tilt angle
        head_tilt = getattr(pose, 'head_tilt_deg', 0)
        payload["angle"] = head_tilt or 0

    return payload


# ── FastAPI Server (runs in its own thread) ──────────────────────────────
def start_api_server():
    """Start the FastAPI local API server."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    api = FastAPI(title="Ghost Stream Local Engine")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class EquipRequest(BaseModel):
        image_url: str
        anchor_type: str = "right_wrist"

    @api.get("/health")
    async def health():
        return {
            "status": "ok" if engine_status["running"] else "starting",
            "service": "ghost-stream-local",
            "fps": engine_status["fps"],
            "prop_count": engine_status["prop_count"],
            "camera": engine_status["camera_resolution"],
        }

    @api.post("/equip")
    async def equip_prop(request: EquipRequest):
        """Receive a prop image URL from the Cloud Backend and hot-swap it."""
        import requests as req
        from starlette.concurrency import run_in_threadpool

        try:
            # Download the prop image
            print(f"[Local API] Downloading prop: {request.image_url}")
            resp = await run_in_threadpool(
                req.get, request.image_url, timeout=30
            )
            resp.raise_for_status()

            # Save to temp file
            import tempfile
            suffix = ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(Path(__file__).parent)) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

            # Process the prop image
            cat = resolve_category(request.anchor_type)
            meta = PropManager.process_image(tmp_path, category=cat)

            if meta is None:
                raise ValueError("Failed to process prop image")

            # Hot-swap: replace all props with the new one
            with props_lock:
                active_props.clear()
                active_props.append({
                    "meta": meta,
                    "scale_filter": _ScaleFilter(),
                    "state": _PropState(),
                })

            print(f"[Local API] Prop equipped: {request.anchor_type}")
            return {"status": "ok", "anchor_type": request.anchor_type}

        except Exception as exc:
            print(f"[Local API] Failed to equip prop: {exc}")
            traceback.print_exc()
            return {"status": "error", "message": str(exc)}

    @api.post("/equip-background")
    async def equip_background(request: EquipRequest):
        """Receive a background image URL and hot-swap it."""
        import requests as req
        from starlette.concurrency import run_in_threadpool

        try:
            print(f"[Local API] Downloading background: {request.image_url}")
            resp = await run_in_threadpool(
                req.get, request.image_url, timeout=30
            )
            resp.raise_for_status()

            # Decode image
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            new_bg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if new_bg is None:
                raise ValueError("Failed to decode background image")

            with bg_lock:
                bg_image_holder["image"] = new_bg

            print("[Local API] Background updated.")
            return {"status": "ok"}

        except Exception as exc:
            print(f"[Local API] Failed to update background: {exc}")
            return {"status": "error", "message": str(exc)}

    @api.post("/clear")
    async def clear_props():
        """Clear all active props."""
        with props_lock:
            active_props.clear()
        print("[Local API] All props cleared.")
        return {"status": "ok"}

    @api.websocket("/ws/anchor")
    async def ws_tracking(websocket: WebSocket):
        """WebSocket endpoint for OBS overlay tracking data.

        Streams tracking data at ~30fps directly to overlay.html
        for minimum latency prop positioning.
        """
        await websocket.accept()
        with ws_clients_lock:
            ws_clients.add(websocket)
        print(f"[WebSocket] Overlay connected. Total: {len(ws_clients)}")

        try:
            while True:
                # Listen for anchor preference updates from overlay
                data = await websocket.receive_text()
                # Could process anchor preferences here if needed
        except WebSocketDisconnect:
            pass
        finally:
            with ws_clients_lock:
                ws_clients.discard(websocket)
            print(f"[WebSocket] Overlay disconnected. Total: {len(ws_clients)}")

    # Run the API server
    config = uvicorn.Config(
        api, host=LOCAL_HOST, port=LOCAL_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server.run()


# ── Heartbeat Thread ─────────────────────────────────────────────────────
def heartbeat_loop():
    """Periodically ping the Cloud Backend with local engine status."""
    import requests as req

    while True:
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)
        if not engine_status["running"]:
            continue
        try:
            req.post(
                f"{CLOUD_API_URL}/heartbeat",
                json=engine_status,
                timeout=3,
            )
        except Exception:
            pass  # Cloud may not be running — that's fine


# ── Entry Point ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Ghost Engine - Live Pipeline")
    ap.add_argument("--props", nargs="+", default=[],
                    help="Prop images. Format: path.png or path.png:category")
    ap.add_argument("--from-pipeline", dest="pipeline_json",
                    help="JSON file with PropRequest list from image-gen pipeline")
    ap.add_argument("--bg", dest="background", default=None,
                    help="AI generated background image path")
    ap.add_argument("--hand", choices=["left", "right", "any"], default="any",
                    help="Default hand/side for props")
    ap.add_argument("--debug", action="store_true",
                    help="Show landmark dots and per-frame diagnostics")
    ap.add_argument("--no-mirror", action="store_true",
                    help="Disable selfie mirror")
    ap.add_argument("--no-api", action="store_true",
                    help="Disable the local API server (standalone mode)")
    args = ap.parse_args()

    print("=" * 60)
    print("  Ghost Stream — Local Vision Engine")
    print("=" * 60)

    if not args.no_api:
        # Start API server in background thread
        print(f"[Ghost Engine] Starting Local API on port {LOCAL_PORT}...")
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()

        # Start heartbeat in background thread
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        # Give the API server a moment to start
        time.sleep(1)
        print(f"[Ghost Engine] Local API ready at http://localhost:{LOCAL_PORT}")
        print(f"[Ghost Engine] WebSocket at ws://localhost:{LOCAL_PORT}/ws/anchor")
    else:
        print("[Ghost Engine] Running in standalone mode (no API server).")

    # Run camera loop on the main thread (OpenCV requires main thread on some OS)
    camera_loop(args)


if __name__ == "__main__":
    main()
