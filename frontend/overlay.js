// Ghost Stream — OBS Overlay Engine
// Connects to:
//   1. Local Engine (ws://localhost:8001/ws/anchor) for 30fps tracking data
//   2. Cloud Backend (ws://<host>/ws/anchor) for new prop/clear events
// 
// This file is CLEAN — no UI elements. The overlay is invisible to viewers.

const LOCAL_WS_URL = 'ws://localhost:8001/ws/anchor';

// Configuration matching the backend python logic
const ZONE_SCALE = {
    "right_wrist":     2.0,
    "left_wrist":      2.0,
    "prop_in_hand":    2.0,
    "both_wrists":     2.5,
    "head":            1.0,
    "left_shoulder":   1.5,
    "right_shoulder":  1.5,
    "both_shoulders":  2.5,
    "ambient":         1.5,
    "background":      1.5,
};

// Fallback grip positions (used when backend doesn't include grip data)
const ZONE_GRIP = {
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
};

// Default prop so the streamer immediately sees something tracking in OBS on startup
const SWORD_SVG = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200' viewBox='0 0 100 100'><text y='80' font-size='80'>🗡️</text></svg>";
let currentProp = {
    url: SWORD_SVG,
    anchorType: "right_wrist",
    gripX: null,
    gripY: null,
    intrinsicAngleDeg: 0,
};

let localWs = null;
let cloudWs = null;
let imgWidth = 0;
let imgHeight = 0;
const propImg = document.getElementById("prop-img");
const bgImg = document.getElementById("bg-img");

// Apply default prop to DOM immediately
propImg.onload = () => {
    imgWidth = propImg.naturalWidth;
    imgHeight = propImg.naturalHeight;
    propImg.style.display = "block";
};
propImg.src = currentProp.url;

// Determine protocols for cloud backend connection
const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

// Standard webcam resolution for coordinate mapping fallback
const VIDEO_W = 640;
const VIDEO_H = 480;

// ── Local Engine WebSocket (Tracking Data - 30fps) ───────────────────────
function connectLocalWs() {
    localWs = new WebSocket(LOCAL_WS_URL);

    localWs.onopen = () => {
        console.log("[Overlay] Connected to Local Engine (tracking data)");
        if (currentProp && currentProp.anchorType) {
            localWs.send(JSON.stringify({ anchor_type: currentProp.anchorType }));
        }
    };

    localWs.onmessage = (event) => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch (err) {
            console.warn("[Overlay] Failed to parse Local WS message:", event.data);
            return;
        }

        // Local engine only sends tracking payloads, not new_prop events
        if (!currentProp || data.error || !data.points || data.points.length === 0) return;

        updatePropPosition(data);
    };

    localWs.onclose = () => {
        console.log("[Overlay] Local Engine WS closed. Reconnecting in 2s...");
        setTimeout(connectLocalWs, 2000);
    };

    localWs.onerror = () => {
        console.log("[Overlay] Local Engine WS error. Will reconnect...");
    };
}

// ── Cloud Backend WebSocket (Prop Events) ────────────────────────────────
function connectCloudWs() {
    const cloudUrl = window.location.protocol === 'file:'
        ? 'ws://localhost:8000/ws/anchor'
        : `${wsProtocol}//${window.location.host}/ws/anchor`;

    cloudWs = new WebSocket(cloudUrl);

    cloudWs.onopen = () => {
        console.log("[Overlay] Connected to Cloud Backend (prop events)");
        if (currentProp && currentProp.anchorType) {
            cloudWs.send(JSON.stringify({ anchor_type: currentProp.anchorType }));
        }
    };

    cloudWs.onmessage = (event) => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch (err) {
            console.warn("[Overlay] Failed to parse Cloud WS message:", event.data);
            return;
        }

        // Cloud backend only sends new_prop events
        if (data.type === "new_prop") {
            handleNewProp(data);
            return;
        }
    };

    cloudWs.onclose = () => {
        console.log("[Overlay] Cloud Backend WS closed. Reconnecting in 3s...");
        setTimeout(connectCloudWs, 3000);
    };

    cloudWs.onerror = () => {
        console.log("[Overlay] Cloud Backend WS error. Will reconnect...");
    };
}

function handleNewProp(data) {
    if (data.action === "clear") {
        currentProp = null;
        propImg.style.display = "none";
        bgImg.style.display = "none";
        return;
    }

    const baseUrl = window.location.protocol === 'file:'
        ? 'http://localhost:8000'
        : `${protocol}//${window.location.host}`;

    currentProp = {
        url: `${baseUrl}/outputs/${data.filename}`,
        anchorType: data.anchor_type,
        gripX: typeof data.grip_x === "number" ? data.grip_x : null,
        gripY: typeof data.grip_y === "number" ? data.grip_y : null,
        intrinsicAngleDeg: typeof data.intrinsic_angle_deg === "number" ? data.intrinsic_angle_deg : 0,
    };

    // Tell local engine what anchor we want tracking for
    if (localWs && localWs.readyState === WebSocket.OPEN) {
        localWs.send(JSON.stringify({ anchor_type: currentProp.anchorType }));
    }

    if (currentProp.anchorType === "background") {
        propImg.style.display = "none";
        // Do NOT show background in HTML, local engine handles it with RVM!
        bgImg.style.display = "none";
    } else {
        bgImg.style.display = "none";
        propImg.onload = () => {
            imgWidth = propImg.naturalWidth;
            imgHeight = propImg.naturalHeight;
            propImg.style.display = "block";

            const startX = (window.innerWidth - imgWidth) / 2;
            const startY = (window.innerHeight - imgHeight) / 2;
            propImg.style.transform = `translate(${startX}px, ${startY}px)`;
        };
        propImg.src = currentProp.url;
    }
}

function updatePropPosition(payload) {
    if (currentProp.anchorType === "background") return;
    if (imgWidth === 0 || imgHeight === 0) return;

    const pt = payload.points[0];

    const intrinsic = currentProp.intrinsicAngleDeg || 0;
    const angle = payload.angle - intrinsic;

    const videoW = payload.frame_width || VIDEO_W;
    const videoH = payload.frame_height || VIDEO_H;
    const scaleX = window.innerWidth / videoW;
    const scaleY = window.innerHeight / videoH;

    const screenX = pt.x * scaleX;
    const screenY = pt.y * scaleY;

    const baseSizeVideoPx = payload.scale * (0.25 * videoW);
    const bodyScalePx = baseSizeVideoPx * ((scaleX + scaleY) / 2);

    const zoneRatio = ZONE_SCALE[currentProp.anchorType] || 1.0;

    const hasCustomGrip = currentProp.gripX !== null && currentProp.gripY !== null;
    const gripXRatio = hasCustomGrip ? currentProp.gripX : 0.5;
    const gripYRatio = hasCustomGrip ? currentProp.gripY : (ZONE_GRIP[currentProp.anchorType] || 0.5);

    const targetSize = bodyScalePx * zoneRatio;
    const scaleFactor = targetSize / Math.max(imgWidth, imgHeight);

    const newW = imgWidth * scaleFactor;
    const newH = imgHeight * scaleFactor;

    const cx = screenX - (newW / 2);
    const cy = screenY - (newH / 2);

    const gripDx = (gripXRatio - 0.5) * newW;
    const gripDy = (gripYRatio - 0.5) * newH;

    const rad = angle * (Math.PI / 180);

    const rotatedGripX = gripDx * Math.cos(rad) + gripDy * Math.sin(rad);
    const rotatedGripY = -gripDx * Math.sin(rad) + gripDy * Math.cos(rad);

    const finalX = cx - rotatedGripX;
    const finalY = cy - rotatedGripY;

    propImg.style.width = `${newW}px`;
    propImg.style.height = `${newH}px`;

    if (payload.brightness !== undefined) {
        propImg.style.filter = `brightness(${payload.brightness})`;
    }

    propImg.style.transform = `translate(${finalX}px, ${finalY}px) rotate(${angle}deg)`;
}

// Start both connections
connectLocalWs();
connectCloudWs();
