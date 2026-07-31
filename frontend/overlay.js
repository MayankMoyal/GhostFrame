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

// Fallback only — used when a broadcast doesn't include grip_x/grip_y
// (e.g. remove_bg was off, so prop_alignment.py never ran). Once a prop
// carries real grip data from the backend, THAT takes priority over this
// table — see handleNewProp()/updatePropPosition().
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
    gripX: null,               // fraction (0..1) within the prop image, or null = use ZONE_GRIP fallback (centerline)
    gripY: null,               // fraction (0..1) within the prop image, or null = use ZONE_GRIP fallback
    intrinsicAngleDeg: 0,      // how far the drawn object deviates from vertical; subtracted from live tracked angle
};

let ws = null;
let imgWidth = 0;
let imgHeight = 0;
const propImg = document.getElementById("prop-img");
const bgImg = document.getElementById("bg-img");

// Apply default prop to DOM immediately
propImg.src = currentProp.url;
propImg.onload = () => {
    imgWidth = propImg.naturalWidth;
    imgHeight = propImg.naturalHeight;
    propImg.style.display = "block";
};

// Determine WS protocol based on current page protocol (used in multiple functions)
const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

// Assume standard webcam resolution for now to map coordinates to screen
const VIDEO_W = 640;
const VIDEO_H = 480;

function connectWebSocket() {
    // Connect to the LOCAL ghost_engine bridge
    ws = new WebSocket(`ws://localhost:8001/ws/anchor`);

    ws.onopen = () => {
        console.log("Overlay connected to WebSocket tracking stream");
        // Tell backend what anchor we want tracking for
        // Send appropriate anchor type on open
        if (currentProp && currentProp.anchorType) {
            ws.send(JSON.stringify({ anchor_type: currentProp.anchorType }));
        } else {
            ws.send(JSON.stringify({ anchor_type: "both_shoulders" }));
        }
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Check if this is a "new_prop" broadcast from the dashboard
        if (data.type === "new_prop") {
            handleNewProp(data);
            return;
        }

        // Otherwise, it's a tracking payload at 30fps
        if (!currentProp || data.error || !data.points || data.points.length === 0) return;

        updatePropPosition(data);
    };

    ws.onclose = () => {
        console.log("WebSocket closed. Reconnecting in 2s...");
        setTimeout(connectWebSocket, 2000);
    };
}

function handleNewProp(data) {
    if (data.action === "clear") {
        currentProp = null;
        propImg.style.display = "none";
        bgImg.style.display = "none";
        return;
    }

    currentProp = {
        url: `${protocol}//${window.location.host}/outputs/${data.filename}`,
        anchorType: data.anchor_type,
        // grip_x/grip_y/intrinsic_angle_deg only exist when prop_alignment.py
        // ran on the backend (remove_bg was on and the crop was valid).
        // Treat anything else as "no data" rather than trusting a stray 0.
        gripX: typeof data.grip_x === "number" ? data.grip_x : null,
        gripY: typeof data.grip_y === "number" ? data.grip_y : null,
        intrinsicAngleDeg: typeof data.intrinsic_angle_deg === "number" ? data.intrinsic_angle_deg : 0,
    };

    // Update websocket tracking target
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ anchor_type: currentProp.anchorType }));
    }

    if (currentProp.anchorType === "background") {
        propImg.style.display = "none";
        bgImg.src = currentProp.url;
        bgImg.style.display = "block";
        // Full-bleed backdrop for OBS.
        bgImg.style.position = "absolute";
        bgImg.style.top = "0";
        bgImg.style.left = "0";
        bgImg.style.transform = "none";
        bgImg.style.width = "100%";
        bgImg.style.height = "100%";
        bgImg.style.objectFit = "cover";
    } else {
        bgImg.style.display = "none";
        propImg.src = currentProp.url;
        propImg.onload = () => {
            imgWidth = propImg.naturalWidth;
            imgHeight = propImg.naturalHeight;
            propImg.style.display = "block";
            
            // TEMPORARY TEST FIX: Center the prop on screen initially
            // This will be overwritten by updatePropPosition() once the webcam sees you
            const startX = (window.innerWidth - imgWidth) / 2;
            const startY = (window.innerHeight - imgHeight) / 2;
            propImg.style.transform = `translate(${startX}px, ${startY}px)`;
        };
    }
}

function updatePropPosition(payload) {
    if (currentProp.anchorType === "background") return; // Handled statically
    if (imgWidth === 0 || imgHeight === 0) return;

    const pt = payload.points[0];
    
    // Subtract the intrinsic drawn angle of the prop from the live tracked angle 
    // so that objects drawn at a slant (e.g. 45deg sword) point straight along the arm.
    const intrinsic = currentProp.intrinsicAngleDeg || 0;
    const angle = payload.angle - intrinsic;

    // Map webcam coordinates to current screen size using dynamic frame dimensions
    const videoW = payload.frame_width || VIDEO_W;
    const videoH = payload.frame_height || VIDEO_H;
    const scaleX = window.innerWidth / videoW;
    const scaleY = window.innerHeight / videoH;
    
    const screenX = pt.x * scaleX;
    const screenY = pt.y * scaleY;
    
    // payload.scale is a multiplier where 1.0 means shoulders are 25% of the frame width.
    // Convert this back to a pixel reference size based on the video resolution.
    const baseSizeVideoPx = payload.scale * (0.25 * videoW);
    
    // Scale body size by same window ratio (average of X and Y scale)
    const bodyScalePx = baseSizeVideoPx * ((scaleX + scaleY) / 2);

    const zoneRatio = ZONE_SCALE[currentProp.anchorType] || 1.0;

    // Prefer the real per-prop grip from prop_alignment.py (computed from
    // the object's own alpha pixels) over the static ZONE_GRIP fraction,
    // which assumes a full-bleed canvas. gripXRatio defaults to 0.5
    // (centerline) same as the old behavior did implicitly.
    const hasCustomGrip = currentProp.gripX !== null && currentProp.gripY !== null;
    const gripXRatio = hasCustomGrip ? currentProp.gripX : 0.5;
    const gripYRatio = hasCustomGrip ? currentProp.gripY : (ZONE_GRIP[currentProp.anchorType] || 0.5);

    const targetSize = bodyScalePx * zoneRatio;
    const scaleFactor = targetSize / Math.max(imgWidth, imgHeight);

    const newW = imgWidth * scaleFactor;
    const newH = imgHeight * scaleFactor;

    // Center offset
    const cx = screenX - (newW / 2);
    const cy = screenY - (newH / 2);

    // Grip offset from image center, in BOTH axes — a PCA-derived grip can
    // sit off the vertical centerline (e.g. a sword drawn at a slant), so
    // this is a full 2D offset, not just the vertical-only offset the
    // static-ZONE_GRIP version used.
    const gripDx = (gripXRatio - 0.5) * newW;
    const gripDy = (gripYRatio - 0.5) * newH;

    // CSS rotation is clockwise. Angle from backend: 0 is UP, CW positive.
    const rad = angle * (Math.PI / 180);

    // Rotate the (gripDx, gripDy) offset vector by `angle` (clockwise),
    // same convention as the original vertical-only rotation.
    const rotatedGripX = gripDx * Math.cos(rad) + gripDy * Math.sin(rad);
    const rotatedGripY = -gripDx * Math.sin(rad) + gripDy * Math.cos(rad);

    // Subtract the grip offset so the designated point lands on the tracking point
    const finalX = cx - rotatedGripX;
    const finalY = cy - rotatedGripY;

    // Apply via CSS transform for max performance
    propImg.style.width = `${newW}px`;
    propImg.style.height = `${newH}px`;
    
    // Lighting
    if (payload.brightness !== undefined) {
        propImg.style.filter = `brightness(${payload.brightness})`;
    }

    propImg.style.transform = `translate(${finalX}px, ${finalY}px) rotate(${angle}deg)`;
}

// Start connection
connectWebSocket();
