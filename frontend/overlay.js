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

let currentProp = null; // { url, anchorType }
let ws = null;
let imgWidth = 0;
let imgHeight = 0;
const propImg = document.getElementById("prop-img");
const bgImg = document.getElementById("bg-img");

// Determine WS protocol based on current page protocol (used in multiple functions)
const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

// Assume standard webcam resolution for now to map coordinates to screen
const VIDEO_W = 640;
const VIDEO_H = 480;

function connectWebSocket() {
    // Connect to the backend
    ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/anchor`);

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
        anchorType: data.anchor_type
    };

    // Update websocket tracking target
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ anchor_type: currentProp.anchorType }));
    }

    if (currentProp.anchorType === "background") {
        propImg.style.display = "none";
        bgImg.src = currentProp.url;
        bgImg.style.display = "block";
        // TEMPORARY TEST FIX: Force it to center so we can see it
        bgImg.style.position = "absolute";
        bgImg.style.top = "50%";
        bgImg.style.left = "50%";
        bgImg.style.transform = "translate(-50%, -50%)";
        bgImg.style.width = "500px"; // temporary fixed width
        bgImg.style.height = "auto";
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
    const angle = payload.angle;
    
    // Calculate mapping from video resolution to current window size
    const scaleX = window.innerWidth / VIDEO_W;
    const scaleY = window.innerHeight / VIDEO_H;
    
    const screenX = pt.x * scaleX;
    const screenY = pt.y * scaleY;
    
    // Scale body size by same window ratio (average of X and Y scale)
    const bodyScalePx = payload.scale * ((scaleX + scaleY) / 2);

    const zoneRatio = ZONE_SCALE[currentProp.anchorType] || 1.0;
    const gripRatio = ZONE_GRIP[currentProp.anchorType] || 0.5;

    const targetSize = bodyScalePx * zoneRatio;
    const scaleFactor = targetSize / Math.max(imgWidth, imgHeight);

    const newW = imgWidth * scaleFactor;
    const newH = imgHeight * scaleFactor;

    // Center offset
    const cx = screenX - (newW / 2);
    const cy = screenY - (newH / 2);

    // Grip offset calculation (rotating the grip point)
    // Positive dy means grip is BELOW center.
    const gripDy = (gripRatio - 0.5) * newH;
    
    // CSS rotation is clockwise. Angle from backend: 0 is UP, CW positive.
    const rad = angle * (Math.PI / 180);
    
    // Vector rotation of the grip offset
    const rotatedGripX = gripDy * Math.sin(rad);
    const rotatedGripY = gripDy * Math.cos(rad);

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
