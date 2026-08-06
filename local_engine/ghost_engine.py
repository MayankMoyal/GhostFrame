"""Ghost Engine — The master pipeline for Ghost Stream (Local PC)."""
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
    CAMERA_WIDTH = 640   # FORCED 640 FOR SPEED
    CAMERA_HEIGHT = 480  # FORCED 480 FOR SPEED
    MIRROR_FEED = True
    HEARTBEAT_INTERVAL_SECONDS = 5

latest_tracking_data = {}
tracking_lock = threading.Lock()
active_props = []
props_lock = threading.Lock()
bg_image_holder = {"image": None}
bg_lock = threading.Lock()
ws_clients = set()
ws_clients_lock = threading.Lock()
engine_status = {"running": False, "fps": 0.0, "prop_count": 0, "camera_resolution": f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"}

async def _ws_broadcast(message: str):
    dead = set()
    with ws_clients_lock: clients = set(ws_clients)
    for ws in clients:
        try: await ws.send_text(message)
        except Exception: dead.add(ws)
    if dead:
        with ws_clients_lock: ws_clients.difference_update(dead)

def broadcast_tracking_sync(data: dict):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running(): asyncio.run_coroutine_threadsafe(_ws_broadcast(json.dumps(data)), loop)
    except RuntimeError: pass

def camera_loop(args):
    global engine_status
    with props_lock:
        if args.pipeline_json:
            for req in load_pipeline_json(args.pipeline_json):
                cat = resolve_category(req.category)
                side = req.target_side or args.hand
                meta = PropManager.process_image(req.image_path, category=cat, prompt=req.prompt, target_side=side)
                if meta: active_props.append({"meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()})
        for spec in args.props:
            if ":" in spec:
                path, cat_str = spec.rsplit(":", 1)
                cat = resolve_category(cat_str)
            else: path, cat = spec, None
            meta = PropManager.process_image(path, category=cat, target_side=args.hand)
            if meta: active_props.append({"meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()})

    with bg_lock:
        if args.background:
            bg_image_holder["image"] = cv2.imread(args.background)
            if bg_image_holder["image"] is None:
                bg_image_holder["image"] = np.full((480, 640, 3), 40, dtype=np.uint8)
        else:
            bg_image_holder["image"] = np.full((480, 640, 3), 40, dtype=np.uint8)

    print("[Ghost Engine] Initializing MediaPipe Tracker...")
    tracker = UltimateTracker()
    print("[Ghost Engine] Initializing RVM Background Remover...")
    remover = BackgroundRemover()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    mirror = MIRROR_FEED and not args.no_mirror
    engine_status["running"] = True
    fps_t, fps_n, fps_val = time.time(), 0, 0.0

    while True:
        ok, raw = cap.read()
        if not ok: break
        
        h, w = raw.shape[:2]
        # Standard horizontal mirror for natural webcam interaction
        display_frame = cv2.flip(raw, 1) if mirror else raw.copy()
        track_result = tracker.detect(raw)

        with bg_lock: current_bg = cv2.resize(bg_image_holder["image"], (w, h))
        with props_lock:
            back_props = [p for p in active_props if p["meta"].profile.z_index < 0]
            front_props = [p for p in active_props if p["meta"].profile.z_index >= 0]

        for pdata in back_props: current_bg = render_prop(current_bg, pdata, track_result, mirror, w, h, args.debug)
        comp = remover.remove(display_frame, current_bg)
        for pdata in front_props: comp = render_prop(comp, pdata, track_result, mirror, w, h, args.debug)

        tracking_payload = _build_tracking_payload(track_result, w, h)
        with tracking_lock: latest_tracking_data.update(tracking_payload)
        broadcast_tracking_sync(tracking_payload)

        fps_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            fps_val = fps_n / elapsed; fps_n = 0; fps_t = time.time()
            engine_status["fps"] = round(fps_val, 1)
            engine_status["prop_count"] = len(active_props)

        cv2.imshow("Ghost Stream", comp)
        if cv2.waitKey(1) & 0xFF == 27: break

    engine_status["running"] = False
    cap.release()
    remover.close()
    cv2.destroyAllWindows()

def _build_tracking_payload(track_result, frame_w, frame_h) -> dict:
    payload = {"frame_width": frame_w, "frame_height": frame_h, "angle": 0, "scale": 1.0, "brightness": 1.0, "points": []}
    if track_result is None or track_result.pose is None: return payload
    pose = track_result.pose
    if pose.shoulder_width and pose.shoulder_width > 0: payload["scale"] = pose.shoulder_width / 0.25
    points = []
    if track_result.hands:
        for hand in track_result.hands:
            points.append({"x": hand.palm[0] * frame_w, "y": hand.palm[1] * frame_h, "side": hand.handedness})
    if pose.head_center: points.append({"x": pose.head_center[0] * frame_w, "y": pose.head_center[1] * frame_h, "zone": "head"})
    if pose.neck_point: points.append({"x": pose.neck_point[0] * frame_w, "y": pose.neck_point[1] * frame_h, "zone": "both_shoulders"})
    payload["points"] = points
    payload["angle"] = pose.head_angle or 0
    return payload

def start_api_server():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    api = FastAPI(title="Ghost Stream Local Engine")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

    class EquipRequest(BaseModel):
        image_url: str
        anchor_type: str = "right_wrist"

    @api.get("/health")
    async def health(): return {"status": "ok" if engine_status["running"] else "starting", "fps": engine_status["fps"], "prop_count": engine_status["prop_count"]}

    @api.post("/equip")
    async def equip_prop(request: EquipRequest):
        import requests as req
        from starlette.concurrency import run_in_threadpool
        try:
            resp = await run_in_threadpool(req.get, request.image_url, timeout=30)
            resp.raise_for_status()
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=str(Path(__file__).parent)) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            cat = resolve_category(request.anchor_type)
            meta = PropManager.process_image(tmp_path, category=cat)
            if meta is None: raise ValueError("Failed to process prop image")
            with props_lock:
                active_props.clear()
                active_props.append({"meta": meta, "scale_filter": _ScaleFilter(), "state": _PropState()})
            return {"status": "ok", "anchor_type": request.anchor_type}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @api.post("/equip-background")
    async def equip_background(request: EquipRequest):
        import requests as req
        from starlette.concurrency import run_in_threadpool
        try:
            resp = await run_in_threadpool(req.get, request.image_url, timeout=30)
            resp.raise_for_status()
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            new_bg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if new_bg is None: raise ValueError("Failed to decode background image")
            with bg_lock: bg_image_holder["image"] = new_bg
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @api.post("/clear")
    async def clear_props():
        with props_lock: active_props.clear()
        return {"status": "ok"}

    @api.websocket("/ws/anchor")
    async def ws_tracking(websocket: WebSocket):
        await websocket.accept()
        with ws_clients_lock: ws_clients.add(websocket)
        try:
            while True: await websocket.receive_text()
        except WebSocketDisconnect: pass
        finally:
            with ws_clients_lock: ws_clients.discard(websocket)

    config = uvicorn.Config(api, host=LOCAL_HOST, port=LOCAL_PORT, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

def heartbeat_loop():
    import requests as req
    while True:
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)
        if not engine_status["running"]: continue
        try: req.post(f"{CLOUD_API_URL}/heartbeat", json=engine_status, timeout=3)
        except Exception: pass

def main():
    ap = argparse.ArgumentParser(description="Ghost Engine - Live Pipeline")
    ap.add_argument("--props", nargs="+", default=[])
    ap.add_argument("--from-pipeline", dest="pipeline_json")
    ap.add_argument("--bg", dest="background", default=None)
    ap.add_argument("--hand", choices=["left", "right", "any"], default="any")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-api", action="store_true")
    args = ap.parse_args()

    if not args.no_api:
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        time.sleep(1)

    camera_loop(args)

if __name__ == "__main__":
    main()