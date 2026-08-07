// Ghost Frame — Unified Dashboard Controller
// Handles: Voice prompts, Text prompts, Custom prop uploads, OBS overlay clearing,
//          Local Engine health monitoring

// Default to localhost if opening the file directly, else use the host
const BACKEND_BASE_URL = window.location.protocol === 'file:'
    ? 'http://localhost:8000'
    : window.location.origin;
const LOCAL_ENGINE_URL = 'http://localhost:8001';

// ── Push to Local Engine (Browser-side relay) ────────────────────────────
// Instead of relying on Cloud→Local reverse tunnel, the browser
// (which is on the same machine as the Local Engine) forwards directly.
async function pushToLocalEngine(filename, anchorType) {
    const imageUrl = `${BACKEND_BASE_URL}/outputs/${filename}`;
    const endpoint = anchorType === 'background'
        ? `${LOCAL_ENGINE_URL}/equip-background`
        : `${LOCAL_ENGINE_URL}/equip`;
    try {
        const resp = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_url: imageUrl, anchor_type: anchorType }),
        });
        const result = await resp.json();
        console.log(`[Dashboard] Pushed ${anchorType} to Local Engine:`, result);
    } catch (err) {
        console.warn('[Dashboard] Could not reach Local Engine:', err);
    }
}

// ── Utility Functions ────────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    document.getElementById("date").textContent = now.toLocaleDateString();
    document.getElementById("time").textContent = now.toLocaleTimeString();
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function setPipelineStep(id, value, active = false) {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = value;
    element.classList.toggle("active", active);
}

function setPreviewLoading(message) {
    const previewBox = document.querySelector(".preview-box");
    previewBox.innerHTML = "";

    const placeholder = document.createElement("div");
    placeholder.className = "preview-placeholder";

    const spinner = document.createElement("div");
    spinner.className = "loading-spinner";

    const title = document.createElement("p");
    title.textContent = message;

    const subtitle = document.createElement("span");
    subtitle.textContent = "Z-Image-Turbo is generating your image.";

    placeholder.append(spinner, title, subtitle);
    previewBox.appendChild(placeholder);
}

function setPreviewError(message) {
    const previewBox = document.querySelector(".preview-box");
    previewBox.innerHTML = "";

    const placeholder = document.createElement("div");
    placeholder.className = "preview-placeholder error-message";

    const title = document.createElement("p");
    title.textContent = "Generation failed";

    const subtitle = document.createElement("span");
    subtitle.textContent = message;

    placeholder.append(title, subtitle);
    previewBox.appendChild(placeholder);
}

function setPreviewImage(imageUrl, promptText) {
    const previewBox = document.querySelector(".preview-box");
    previewBox.innerHTML = "";

    const image = document.createElement("img");
    image.src = imageUrl;
    image.alt = promptText;
    image.className = "generated-image";
    image.onerror = function () {
        setPreviewError("The backend returned a filename, but the image could not be loaded.");
    };

    previewBox.appendChild(image);
}

function resetPreview() {
    document.querySelector(".preview-box").innerHTML = `
        <div class="preview-placeholder">
            <p>No image generated yet</p>
            <span>Your generated image will appear here.</span>
        </div>
    `;
}

function resetPipeline() {
    setPipelineStep("step1", "Waiting for prompt");
    setPipelineStep("step2", "Model ready");
    setPipelineStep("step3", "Idle");
    setPipelineStep("step4", "Image not rendered");
    setPipelineStep("step5", "Ready");
}

// ── Recent Generations ──────────────────────────────────────────────────
const recentGenerations = [];

function addToRecentGenerations(imageUrl, promptText) {
    recentGenerations.unshift({ imageUrl, promptText });
    if (recentGenerations.length > 4) {
        recentGenerations.pop();
    }
    renderRecentGenerations();
}

function renderRecentGenerations() {
    const container = document.getElementById("recentImages");
    container.innerHTML = "";

    if (recentGenerations.length === 0) {
        container.innerHTML = `<p class="empty-gallery">No generations yet — your images will appear here.</p>`;
        return;
    }

    recentGenerations.forEach(item => {
        const card = document.createElement("div");
        card.className = "image-card";

        const img = document.createElement("img");
        img.src = item.imageUrl;
        img.alt = item.promptText;
        img.title = item.promptText;

        card.appendChild(img);
        container.appendChild(card);
    });
}

// ── Text Prompt Generation ──────────────────────────────────────────────
async function generateImage() {
    const generateBtn = document.getElementById("generate");
    const promptInput = document.getElementById("prompt");
    const styleSelect = document.getElementById("style");

    const prompt = promptInput.value.trim();
    const style = styleSelect.value.trim();

    if (!prompt) {
        setText("analysis", "Prompt cannot be empty.");
        setPreviewError("Please enter an image prompt before generating.");
        return;
    }

    generateBtn.textContent = "Generating...";
    generateBtn.disabled = true;

    setText("analysis", "Processing");
    setText("sceneType", "Detecting");
    setText("imageStyle", style || "-");
    setText("lighting", "Analyzing");
    setText("timeEstimate", "Running");
    setText("vramUsage", "Measuring");

    setPipelineStep("step1", "Prompt received", true);
    setPipelineStep("step2", "Z-Image-Turbo loaded", true);
    setPipelineStep("step3", "Running inference", true);
    setPipelineStep("step4", "Rendering image", true);
    setPipelineStep("step5", "Waiting for output");
    setPreviewLoading("Generating image...");

    try {
        const response = await fetch(`${BACKEND_BASE_URL}/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, style }),
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(data.message || data.detail || `HTTP ${response.status}`);
        }

        if (!data.filename) {
            throw new Error("The backend did not return an output filename.");
        }

        const imageUrl = `${BACKEND_BASE_URL}/outputs/${data.filename}`;
        setPreviewImage(imageUrl, prompt);
        addToRecentGenerations(imageUrl, prompt);

        setText("analysis", data.agent?.final_prompt || "Completed");
        setText("sceneType", `Anchor: ${data.agent?.anchor_type || "background"}`);
        setText("imageStyle", data.agent?.style_detected || style);
        setText("lighting", "Turbo guidance");
        setText("timeEstimate", data.metrics?.latency_seconds ? `${data.metrics.latency_seconds} sec` : "Complete");
        setText("vramUsage", data.metrics?.peak_vram_gb ? `${data.metrics.peak_vram_gb} GB` : "-");

        setPipelineStep("step4", "Image rendered", true);
        setPipelineStep("step5", "Sent to OBS", true);

        // Forward to Local Engine directly (no reverse tunnel needed)
        const anchorType = data.agent?.anchor_type || 'background';
        pushToLocalEngine(data.filename, anchorType);
    } catch (error) {
        setText("analysis", error.message);
        setPipelineStep("step5", "Error");
        setPreviewError(error.message);
    } finally {
        generateBtn.textContent = "Generate Image";
        generateBtn.disabled = false;
    }
}

// ── Voice Prompt Generation ─────────────────────────────────────────────
async function sendVoicePrompt(audioBlob) {
    const styleSelect = document.getElementById("style");
    const style = styleSelect.value.trim();

    setText("analysis", "Processing Audio");
    setText("sceneType", "Detecting");
    setText("imageStyle", style || "-");
    setText("lighting", "Analyzing");
    setText("timeEstimate", "Running");
    setText("vramUsage", "Measuring");

    setPipelineStep("step1", "Audio received", true);
    setPipelineStep("step2", "Transcribing & Loading Model", true);
    setPipelineStep("step3", "Running inference", true);
    setPipelineStep("step4", "Rendering image", true);
    setPipelineStep("step5", "Waiting for output");
    setPreviewLoading("Generating image from voice...");

    try {
        const formData = new FormData();
        formData.append("audio", audioBlob, "voice_prompt.webm");
        formData.append("style", style);
        formData.append("remove_bg", "true");

        const response = await fetch(`${BACKEND_BASE_URL}/generate-voice`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(data.message || data.detail || `HTTP ${response.status}`);
        }

        if (!data.filename) {
            throw new Error("The backend did not return an output filename.");
        }

        const promptText = data.transcript || "Voice Prompt";
        const imageUrl = `${BACKEND_BASE_URL}/outputs/${data.filename}`;
        setPreviewImage(imageUrl, promptText);
        addToRecentGenerations(imageUrl, promptText);

        micStatus.textContent = "Ready";
        micStatus.style.color = "#4ADE80";
        setText("analysis", `Transcript: "${promptText}"`);
        setText("sceneType", `Anchor: ${data.agent?.anchor_type || "background"}`);
        setText("lighting", "Turbo guidance");
        setText("timeEstimate", data.metrics?.latency_seconds ? `${data.metrics.latency_seconds} sec` : "Complete");
        setText("vramUsage", data.metrics?.peak_vram_gb ? `${data.metrics.peak_vram_gb} GB` : "-");

        setPipelineStep("step4", "Image rendered", true);
        setPipelineStep("step5", "Sent to OBS Overlay", true);

        // Forward to Local Engine directly (no reverse tunnel needed)
        const anchorType = data.agent?.anchor_type || 'background';
        pushToLocalEngine(data.filename, anchorType);
    } catch (error) {
        setText("analysis", error.message);
        setPipelineStep("step5", "Error");
        setPreviewError(error.message);
    }
}

// ── Media Recorder (Push-to-Talk) ───────────────────────────────────────
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
const micBtn = document.getElementById("micBtn");
const micStatus = document.getElementById("micStatus");

async function setupAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const options = MediaRecorder.isTypeSupported('audio/webm')
            ? { mimeType: 'audio/webm' }
            : undefined;

        mediaRecorder = new MediaRecorder(stream, options);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            sendVoicePrompt(audioBlob);
            audioChunks = [];
        };
    } catch (err) {
        console.error("Microphone access denied:", err);
        micStatus.textContent = "Microphone access denied!";
        micStatus.style.color = "red";
    }
}

function startRecording() {
    if (!mediaRecorder || isRecording) return;
    isRecording = true;
    audioChunks = [];
    mediaRecorder.start();

    micBtn.style.background = "#ef4444";
    micBtn.textContent = "🎙️ Recording... Release to send";
    micStatus.textContent = "Listening...";
}

function stopRecording() {
    if (!mediaRecorder || !isRecording) return;
    isRecording = false;
    mediaRecorder.stop();

    micBtn.style.background = "";
    micBtn.textContent = "🎤 Hold Space or Click to Speak";
    micStatus.textContent = "Processing...";
}

setupAudio();

micBtn.addEventListener("mousedown", startRecording);
micBtn.addEventListener("mouseup", stopRecording);
micBtn.addEventListener("mouseleave", stopRecording);

window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && document.activeElement.tagName !== 'TEXTAREA' && document.activeElement.tagName !== 'SELECT') {
        e.preventDefault();
        startRecording();
    }
});
window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
        stopRecording();
    }
});

// ── Custom Prop Upload ──────────────────────────────────────────────────
const propUploadBtn = document.getElementById("propUploadBtn");
const propUploadInput = document.getElementById("propUploadInput");
const uploadAnchorSelect = document.getElementById("uploadAnchor");

if (propUploadBtn && propUploadInput) {
    propUploadBtn.addEventListener("click", () => propUploadInput.click());

    propUploadInput.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const anchorType = uploadAnchorSelect.value;

        setText("analysis", `Uploading custom prop...`);
        setText("sceneType", `Anchor: ${anchorType}`);
        setPreviewLoading("Uploading and processing custom prop...");

        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("anchor_type", anchorType);

            const response = await fetch(`${BACKEND_BASE_URL}/upload-prop`, {
                method: "POST",
                body: formData,
            });

            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(data.message || data.detail || `HTTP ${response.status}`);
            }

            const imageUrl = `${BACKEND_BASE_URL}/outputs/${data.filename}`;
            setPreviewImage(imageUrl, "Custom Uploaded Prop");
            addToRecentGenerations(imageUrl, "Custom Upload");

            setText("analysis", "Custom prop processed and broadcast successfully.");
        } catch (error) {
            setText("analysis", error.message);
            setPreviewError(error.message);
        } finally {
            propUploadInput.value = "";
        }
    });
}

// ── Clear Props ─────────────────────────────────────────────────────────
document.getElementById("clearPreview").addEventListener("click", () => {
    resetPreview();
    fetch(`${BACKEND_BASE_URL}/clear-props`, { method: "POST" }).catch(console.error);
});

// ── Generate Button ─────────────────────────────────────────────────────
document.getElementById("generate").addEventListener("click", generateImage);

// ── Sidebar Toggle ──────────────────────────────────────────────────────
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");

if (sidebar && sidebarToggle) {
    sidebarToggle.addEventListener("click", function () {
        sidebar.classList.toggle("collapsed");
    });
}

// ── Local Engine Health Check ───────────────────────────────────────────
const engineDot = document.getElementById("engineDot");
const engineStatusEl = document.getElementById("engineStatus");

async function checkEngineHealth() {
    try {
        const resp = await fetch(`${LOCAL_ENGINE_URL}/health`, { signal: AbortSignal.timeout(3000) });
        const data = await resp.json();
        if (data.status === "ok") {
            engineDot.className = "status-dot online";
            engineStatusEl.textContent = `Local Engine: 🟢 ${data.fps !== undefined ? data.fps + ' FPS' : 'Online'}`;
        } else {
            engineDot.className = "status-dot starting";
            engineStatusEl.textContent = "Local Engine: 🟡 Starting...";
        }
    } catch {
        engineDot.className = "status-dot offline";
        engineStatusEl.textContent = "Local Engine: 🔴 Offline";
    }
}

// ── Initialize ──────────────────────────────────────────────────────────
updateClock();
setInterval(updateClock, 1000);
resetPipeline();
checkEngineHealth();
setInterval(checkEngineHealth, 5000);

// ── WebSocket Listener (Sync with OBS Panel) ────────────────────────────
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsHost = window.location.protocol === 'file:' ? 'localhost:8000' : window.location.host;
const dashboardWs = new WebSocket(`${wsProtocol}//${wsHost}/ws/anchor`);

dashboardWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "new_prop") {
        const imageUrl = `${BACKEND_BASE_URL}/outputs/${data.filename}`;
        const prompt = data.agent?.original_prompt || "Generated via OBS Panel";
        
        // Update Preview & Gallery
        setPreviewImage(imageUrl, prompt);
        addToRecentGenerations(imageUrl, prompt);

        // Update Statistics if available
        setText("analysis", data.agent?.final_prompt || "Completed via OBS Panel");
        setText("sceneType", `Anchor: ${data.anchor_type || "background"}`);
        setText("imageStyle", data.agent?.style || "-");
        setText("lighting", "Turbo guidance");
        
        if (data.metrics) {
            setText("timeEstimate", `${data.metrics.latency_seconds} sec`);
            setText("vramUsage", `${data.metrics.peak_vram_gb} GB`);
        } else {
            setText("timeEstimate", "Complete");
            setText("vramUsage", "-");
        }

        setPipelineStep("step4", "Image rendered", true);
        setPipelineStep("step5", "Sent to OBS", true);
    }
};

dashboardWs.onerror = () => console.log("[Dashboard] WebSocket error. Stats won't sync.");
