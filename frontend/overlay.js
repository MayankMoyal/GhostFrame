// ═══════════════════════════════════════════════════════════════════════════
// Ghost Stream — OBS Overlay Engine + Game Bar Panel
// ═══════════════════════════════════════════════════════════════════════════
//
// This file handles:
//   1. Prop tracking via WebSocket to Local Engine (30fps body tracking)
//   2. Cloud WebSocket for prop change events
//   3. Game Bar Panel UI (toggle, tabs, text input, voice input)
//   4. Text generation → POST /generate
//   5. Voice generation → POST /generate-voice (MediaRecorder + Whisper STT)
//   6. Browser-side relay to Local Engine (/equip, /equip-background)
//   7. Status monitoring for Cloud Backend + Local Engine
//
// ═══════════════════════════════════════════════════════════════════════════

// ── Configuration ────────────────────────────────────────────────────────
const LOCAL_WS_URL   = 'ws://localhost:8001/ws/anchor';
const LOCAL_API_URL  = 'http://localhost:8001';

// Auto-detect backend URL: if opened as file://, use localhost; else use same origin
const BACKEND_BASE_URL = window.location.protocol === 'file:'
    ? 'http://localhost:8000'
    : window.location.origin;

const protocol   = window.location.protocol === 'https:' ? 'https:' : 'http:';
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

// Standard webcam resolution for coordinate mapping fallback
const VIDEO_W = 640;
const VIDEO_H = 480;

// Zone scales matching the backend anchor logic
const ZONE_SCALE = {
    "right_wrist":     2.0,
    "left_wrist":      2.0,
    "prop_in_hand":    2.0,
    "hand_held":       2.0,
    "shield":          2.0,
    "both_wrists":     2.5,
    "wrist_wear":      2.0,
    "head":            1.0,
    "head_wear":       1.0,
    "neck_wear":       1.2,
    "ear_wear":        1.0,
    "face_wear":       1.0,
    "left_shoulder":   1.5,
    "right_shoulder":  1.5,
    "both_shoulders":  2.5,
    "body_wear":       2.5,
    "ambient":         1.5,
    "background":      1.5,
};

const ZONE_GRIP = {
    "right_wrist":     0.85,
    "left_wrist":      0.85,
    "prop_in_hand":    0.85,
    "hand_held":       0.85,
    "shield":          0.85,
    "both_wrists":     0.5,
    "wrist_wear":      0.85,
    "head":            0.9,
    "head_wear":       0.9,
    "neck_wear":       0.7,
    "ear_wear":        0.9,
    "face_wear":       0.9,
    "left_shoulder":   0.5,
    "right_shoulder":  0.5,
    "both_shoulders":  0.5,
    "body_wear":       0.5,
    "ambient":         0.5,
    "background":      0.5,
};

// Default prop (sword emoji) so the streamer sees something tracking immediately
const SWORD_SVG = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200' viewBox='0 0 100 100'><text y='80' font-size='80'>🗡️</text></svg>";

let currentProp = {
    url: SWORD_SVG,
    anchorType: "right_wrist",
    gripX: null,
    gripY: null,
    intrinsicAngleDeg: 0,
};

// ── DOM References ───────────────────────────────────────────────────────
const propImg  = document.getElementById("prop-img");
const bgImg    = document.getElementById("bg-img");
let imgWidth   = 0;
let imgHeight  = 0;

// Apply default prop immediately
propImg.src = currentProp.url;
propImg.onload = () => {
    imgWidth  = propImg.naturalWidth;
    imgHeight = propImg.naturalHeight;
    propImg.style.display = "block";
};

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: PROP TRACKING (WebSocket to Local Engine)
// ═══════════════════════════════════════════════════════════════════════════

let localWs = null;
let cloudWs = null;

function connectLocalWs() {
    localWs = new WebSocket(LOCAL_WS_URL);

    localWs.onopen = () => {
        console.log("[Overlay] Connected to Local Engine (tracking data)");
        updateEngineStatus(true);
        if (currentProp && currentProp.anchorType) {
            localWs.send(JSON.stringify({ anchor_type: currentProp.anchorType }));
        }
    };

    localWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (!currentProp || data.error || !data.points || data.points.length === 0) return;
        updatePropPosition(data);
    };

    localWs.onclose = () => {
        console.log("[Overlay] Local Engine WS closed. Reconnecting in 2s...");
        updateEngineStatus(false);
        setTimeout(connectLocalWs, 2000);
    };

    localWs.onerror = () => {
        console.log("[Overlay] Local Engine WS error. Will reconnect...");
        updateEngineStatus(false);
    };
}

function connectCloudWs() {
    const cloudUrl = window.location.protocol === 'file:'
        ? 'ws://localhost:8000/ws/anchor'
        : `${wsProtocol}//${window.location.host}/ws/anchor`;

    cloudWs = new WebSocket(cloudUrl);

    cloudWs.onopen = () => {
        console.log("[Overlay] Connected to Cloud Backend (prop events)");
        updateCloudStatus(true);
        if (currentProp && currentProp.anchorType) {
            cloudWs.send(JSON.stringify({ anchor_type: currentProp.anchorType }));
        }
    };

    cloudWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "new_prop") {
            handleNewProp(data);
        }
    };

    cloudWs.onclose = () => {
        console.log("[Overlay] Cloud Backend WS closed. Reconnecting in 3s...");
        updateCloudStatus(false);
        setTimeout(connectCloudWs, 3000);
    };

    cloudWs.onerror = () => {
        console.log("[Overlay] Cloud Backend WS error. Will reconnect...");
        updateCloudStatus(false);
    };
}

function handleNewProp(data) {
    if (data.action === "clear") {
        currentProp = null;
        propImg.style.display = "none";
        bgImg.style.display   = "none";
        return;
    }

    const baseUrl = window.location.protocol === 'file:'
        ? `http://localhost:8000`
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
        bgImg.style.display   = "none"; // Local engine handles BG with RVM
    } else {
        bgImg.style.display = "none";
        propImg.src = currentProp.url;
        propImg.onload = () => {
            imgWidth  = propImg.naturalWidth;
            imgHeight = propImg.naturalHeight;
            propImg.style.display = "block";
            const startX = (window.innerWidth - imgWidth) / 2;
            const startY = (window.innerHeight - imgHeight) / 2;
            propImg.style.transform = `translate(${startX}px, ${startY}px)`;
        };
    }

    // Update last generated preview in panel
    updateLastGenerated(currentProp.url, data.filename);
}

function updatePropPosition(payload) {
    if (!currentProp || currentProp.anchorType === "background") return;
    if (imgWidth === 0 || imgHeight === 0) return;

    const pt = payload.points[0];
    const intrinsic = currentProp.intrinsicAngleDeg || 0;
    const angle = payload.angle - intrinsic;

    const videoW = payload.frame_width  || VIDEO_W;
    const videoH = payload.frame_height || VIDEO_H;
    const scaleX = window.innerWidth  / videoW;
    const scaleY = window.innerHeight / videoH;

    const screenX = pt.x * scaleX;
    const screenY = pt.y * scaleY;

    const baseSizeVideoPx = payload.scale * (0.25 * videoW);
    const bodyScalePx     = baseSizeVideoPx * ((scaleX + scaleY) / 2);

    const zoneRatio = ZONE_SCALE[currentProp.anchorType] || 1.0;

    const hasCustomGrip = currentProp.gripX !== null && currentProp.gripY !== null;
    const gripXRatio = hasCustomGrip ? currentProp.gripX : 0.5;
    const gripYRatio = hasCustomGrip ? currentProp.gripY : (ZONE_GRIP[currentProp.anchorType] || 0.5);

    const targetSize  = bodyScalePx * zoneRatio;
    const scaleFactor = targetSize / Math.max(imgWidth, imgHeight);

    const newW = imgWidth  * scaleFactor;
    const newH = imgHeight * scaleFactor;

    const cx = screenX - (newW / 2);
    const cy = screenY - (newH / 2);

    const gripDx = (gripXRatio - 0.5) * newW;
    const gripDy = (gripYRatio - 0.5) * newH;

    const rad = angle * (Math.PI / 180);
    const rotatedGripX =  gripDx * Math.cos(rad) + gripDy * Math.sin(rad);
    const rotatedGripY = -gripDx * Math.sin(rad) + gripDy * Math.cos(rad);

    const finalX = cx - rotatedGripX;
    const finalY = cy - rotatedGripY;

    propImg.style.width  = `${newW}px`;
    propImg.style.height = `${newH}px`;

    if (payload.brightness !== undefined) {
        propImg.style.filter = `brightness(${payload.brightness})`;
    }

    propImg.style.transform = `translate(${finalX}px, ${finalY}px) rotate(${angle}deg)`;
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: GAME BAR PANEL UI
// ═══════════════════════════════════════════════════════════════════════════

const panel         = document.getElementById("gamebar-panel");
const trigger       = document.getElementById("gamebar-trigger");
const backdrop      = document.getElementById("panel-backdrop");
const closeBtn      = document.getElementById("panel-close-btn");
const tabBtns       = document.querySelectorAll(".tab-btn");
const tabPanes      = document.querySelectorAll(".tab-pane");

let panelOpen = false;

function openPanel() {
    panelOpen = true;
    panel.classList.add("open");
    backdrop.classList.add("visible");
    trigger.classList.add("hidden");
}

function closePanel() {
    panelOpen = false;
    panel.classList.remove("open");
    backdrop.classList.remove("visible");
    trigger.classList.remove("hidden");
}

function togglePanel() {
    panelOpen ? closePanel() : openPanel();
}

// Trigger button click
trigger.addEventListener("click", openPanel);

// Close button click
closeBtn.addEventListener("click", closePanel);

// Backdrop click closes panel
backdrop.addEventListener("click", closePanel);

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        const tabId = btn.dataset.tab;
        tabBtns.forEach(b => b.classList.remove("active"));
        tabPanes.forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`tab-${tabId}`).classList.add("active");
    });
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
    // Ctrl+G → toggle panel
    if (e.ctrlKey && e.key.toLowerCase() === "g") {
        e.preventDefault();
        togglePanel();
        return;
    }

    // Escape → close panel
    if (e.key === "Escape" && panelOpen) {
        e.preventDefault();
        closePanel();
        return;
    }

    // Space → push-to-talk (only if voice tab is active and not in text input)
    if (e.code === "Space" && panelOpen && !e.repeat) {
        const activeTab = document.querySelector(".tab-pane.active");
        const focusedEl = document.activeElement;
        // Don't trigger if user is typing in textarea
        if (activeTab && activeTab.id === "tab-voice" && focusedEl.tagName !== "TEXTAREA" && focusedEl.tagName !== "INPUT") {
            e.preventDefault();
            startRecording();
        }
    }
});

document.addEventListener("keyup", (e) => {
    if (e.code === "Space" && isRecording) {
        e.preventDefault();
        stopRecording();
    }
});


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: VOICE INPUT (MediaRecorder → /generate-voice)
// ═══════════════════════════════════════════════════════════════════════════

const micBtn      = document.getElementById("mic-btn");
const micLabel    = document.getElementById("mic-label");
const waveformEl  = document.getElementById("waveform");

let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;
let audioStream   = null;
let analyserNode  = null;
let animFrameId   = null;

// Create waveform bars
const NUM_BARS = 24;
for (let i = 0; i < NUM_BARS; i++) {
    const bar = document.createElement("div");
    bar.className = "waveform-bar";
    waveformEl.appendChild(bar);
}
const waveformBars = waveformEl.querySelectorAll(".waveform-bar");

async function startRecording() {
    if (isRecording) return;

    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
        showToast("Microphone access denied", true);
        return;
    }

    isRecording = true;
    audioChunks = [];

    // Setup analyser for waveform visualization
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source   = audioCtx.createMediaStreamSource(audioStream);
    analyserNode   = audioCtx.createAnalyser();
    analyserNode.fftSize = 64;
    source.connect(analyserNode);

    // Start visualizer
    waveformEl.classList.add("active");
    animateWaveform();

    // Setup MediaRecorder
    mediaRecorder = new MediaRecorder(audioStream, { mimeType: "audio/webm" });
    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
        waveformEl.classList.remove("active");
        cancelAnimationFrame(animFrameId);
        resetWaveform();

        // Stop mic stream
        audioStream.getTracks().forEach(t => t.stop());

        if (audioChunks.length === 0) return;

        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        await sendVoiceToBackend(audioBlob);
    };

    mediaRecorder.start();
    micBtn.classList.add("recording");
    micLabel.textContent = "Recording... Release to send";
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    isRecording = false;
    mediaRecorder.stop();
    micBtn.classList.remove("recording");
    micLabel.innerHTML = 'Hold to speak or press <kbd>Space</kbd>';
}

function animateWaveform() {
    if (!analyserNode) return;

    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
    analyserNode.getByteFrequencyData(dataArray);

    const step = Math.floor(dataArray.length / NUM_BARS);
    for (let i = 0; i < NUM_BARS; i++) {
        const val = dataArray[i * step] || 0;
        const height = Math.max(4, (val / 255) * 36);
        waveformBars[i].style.height = `${height}px`;
    }

    animFrameId = requestAnimationFrame(animateWaveform);
}

function resetWaveform() {
    waveformBars.forEach(bar => { bar.style.height = "8px"; });
}

async function sendVoiceToBackend(audioBlob) {
    micLabel.textContent = "Processing voice...";

    const formData = new FormData();
    formData.append("audio", audioBlob, "voice.webm");
    formData.append("style", document.getElementById("voice-style").value);

    try {
        const resp = await fetch(`${BACKEND_BASE_URL}/generate-voice`, {
            method: "POST",
            body: formData,
        });

        const data = await resp.json().catch(() => ({}));

        if (!resp.ok) {
            throw new Error(data.message || data.detail || `HTTP ${resp.status}`);
        }

        const filename = data.filename;
        if (!filename) throw new Error("No output filename returned.");

        // Push to local engine via browser relay
        const anchorType = data.agent?.anchor_type || "right_wrist";
        await pushToLocalEngine(filename, anchorType);

        const imageUrl = `${BACKEND_BASE_URL}/outputs/${filename}`;
        updateLastGenerated(imageUrl, data.agent?.original_prompt || "Voice prompt");

        showToast(`✨ Prop generated: "${data.transcript || 'voice'}"`, false);
        micLabel.innerHTML = 'Hold to speak or press <kbd>Space</kbd>';

        // Auto-close panel after short delay
        setTimeout(closePanel, 1800);

    } catch (err) {
        showToast(err.message, true);
        micLabel.innerHTML = 'Hold to speak or press <kbd>Space</kbd>';
    }
}

// Mouse-based push-to-talk
micBtn.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startRecording();
});
micBtn.addEventListener("mouseup", stopRecording);
micBtn.addEventListener("mouseleave", () => {
    if (isRecording) stopRecording();
});

// Touch support
micBtn.addEventListener("touchstart", (e) => {
    e.preventDefault();
    startRecording();
});
micBtn.addEventListener("touchend", stopRecording);


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: TEXT INPUT (POST /generate)
// ═══════════════════════════════════════════════════════════════════════════

const textPrompt    = document.getElementById("text-prompt");
const textGenBtn    = document.getElementById("text-generate-btn");
const textStyle     = document.getElementById("text-style");

textGenBtn.addEventListener("click", generateFromText);

// Enter + Ctrl to submit
textPrompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
        e.preventDefault();
        generateFromText();
    }
});

async function generateFromText() {
    const prompt = textPrompt.value.trim();
    if (!prompt) {
        showToast("Please enter a prompt", true);
        return;
    }

    textGenBtn.disabled    = true;
    textGenBtn.textContent = "⏳ Generating...";
    textGenBtn.classList.add("loading");

    try {
        const resp = await fetch(`${BACKEND_BASE_URL}/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                style: textStyle.value,
            }),
        });

        const data = await resp.json().catch(() => ({}));

        if (!resp.ok) {
            throw new Error(data.message || data.detail || `HTTP ${resp.status}`);
        }

        const filename = data.filename;
        if (!filename) throw new Error("No output filename returned.");

        // Push to local engine via browser relay
        const anchorType = data.agent?.anchor_type || "right_wrist";
        await pushToLocalEngine(filename, anchorType);

        const imageUrl = `${BACKEND_BASE_URL}/outputs/${filename}`;
        updateLastGenerated(imageUrl, prompt);

        showToast("✨ Prop generated!", false);
        textPrompt.value = "";

        // Auto-close panel
        setTimeout(closePanel, 1800);

    } catch (err) {
        showToast(err.message, true);
    } finally {
        textGenBtn.disabled    = false;
        textGenBtn.textContent = "⚡ Generate Prop";
        textGenBtn.classList.remove("loading");
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5: BROWSER-SIDE RELAY TO LOCAL ENGINE
// ═══════════════════════════════════════════════════════════════════════════

async function pushToLocalEngine(filename, anchorType) {
    const imageUrl = `${BACKEND_BASE_URL}/outputs/${filename}`;
    const endpoint = anchorType === "background"
        ? `${LOCAL_API_URL}/equip-background`
        : `${LOCAL_API_URL}/equip`;

    try {
        const resp = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_url: imageUrl, anchor_type: anchorType }),
        });
        const result = await resp.json();
        console.log(`[Panel] Pushed ${anchorType} to Local Engine:`, result);
    } catch (err) {
        console.warn("[Panel] Could not reach Local Engine:", err);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6: CLEAR PROPS
// ═══════════════════════════════════════════════════════════════════════════

const clearPropsBtn = document.getElementById("clear-props-btn");

clearPropsBtn.addEventListener("click", async () => {
    try {
        await fetch(`${BACKEND_BASE_URL}/clear-props`, { method: "POST" });
        currentProp = null;
        propImg.style.display = "none";
        bgImg.style.display   = "none";
        document.getElementById("last-gen").classList.add("hidden");
        showToast("Props cleared", false);
    } catch (err) {
        showToast("Failed to clear props", true);
    }
});


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: STATUS MONITORING
// ═══════════════════════════════════════════════════════════════════════════

const cloudStatusDot  = document.getElementById("cloud-status-dot");
const cloudStatusText = document.getElementById("cloud-status-text");
const engineStatusDot  = document.getElementById("engine-status-dot");
const engineStatusText = document.getElementById("engine-status-text");

function updateCloudStatus(online) {
    cloudStatusDot.className  = `status-dot ${online ? "online" : "offline"}`;
    cloudStatusText.textContent = online ? "Cloud ✓" : "Cloud ✗";
}

function updateEngineStatus(online) {
    engineStatusDot.className  = `status-dot ${online ? "online" : "offline"}`;
    engineStatusText.textContent = online ? "Engine ✓" : "Engine ✗";
}

// Periodic health check for Cloud Backend
async function checkCloudHealth() {
    try {
        const resp = await fetch(`${BACKEND_BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
        const data = await resp.json();
        updateCloudStatus(data.status === "ok");
    } catch {
        // Don't mark as offline if WS is connected
        if (!cloudWs || cloudWs.readyState !== WebSocket.OPEN) {
            updateCloudStatus(false);
        }
    }
}

// Periodic health check for Local Engine
async function checkEngineHealth() {
    try {
        const resp = await fetch(`${LOCAL_API_URL}/health`, { signal: AbortSignal.timeout(3000) });
        const data = await resp.json();
        updateEngineStatus(data.status === "ok" || resp.ok);
    } catch {
        if (!localWs || localWs.readyState !== WebSocket.OPEN) {
            updateEngineStatus(false);
        }
    }
}

// Check health every 10 seconds
setInterval(() => {
    checkCloudHealth();
    checkEngineHealth();
}, 10000);

// Initial checks
checkCloudHealth();
checkEngineHealth();


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8: TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════════════════

const toast = document.getElementById("toast");
let toastTimer = null;

function showToast(message, isError = false) {
    clearTimeout(toastTimer);
    toast.textContent = message;
    toast.className   = isError ? "error" : "";
    // Force reflow for animation restart
    toast.offsetHeight;
    toast.classList.add("visible");

    toastTimer = setTimeout(() => {
        toast.classList.remove("visible");
    }, 3500);
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9: LAST GENERATED PREVIEW
// ═══════════════════════════════════════════════════════════════════════════

const lastGenEl     = document.getElementById("last-gen");
const lastGenImg    = document.getElementById("last-gen-img");
const lastGenPrompt = document.getElementById("last-gen-prompt");

function updateLastGenerated(imageUrl, promptText) {
    lastGenImg.src = imageUrl;
    lastGenPrompt.textContent = promptText || "Generated prop";
    lastGenEl.classList.remove("hidden");
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 10: INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

// Start WebSocket connections
connectLocalWs();
connectCloudWs();

console.log("[Overlay] GhostFrame Game Bar Overlay initialized");
console.log("[Overlay] Press Ctrl+G to toggle the control panel");
