// Default to localhost if opening the file directly, else use the host
const BACKEND_BASE_URL = window.location.protocol === 'file:' ? 'http://localhost:8000' : window.location.origin;

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
    if (!element) {
        return;
    }

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

        // The API returns the transcript, we can show it
        const promptText = data.transcript || "Voice Prompt";

        const imageUrl = `${BACKEND_BASE_URL}/outputs/${data.filename_nobg || data.filename}`;
        setPreviewImage(imageUrl, promptText);
        addToRecentGenerations(imageUrl, promptText);

        // Reset mic status to ready after image is shown
        micStatus.textContent = "Ready";
        micStatus.style.color = "#4ADE80"; // green
        setText("analysis", `Transcript: "${promptText}"`);
        setText("sceneType", `Anchor: ${data.agent?.anchor_type || "background"}`);
        setText("lighting", "Turbo guidance");
        setText(
            "timeEstimate",
            data.metrics?.latency_seconds ? `${data.metrics.latency_seconds} sec` : "Complete"
        );
        setText(
            "vramUsage",
            data.metrics?.peak_vram_gb ? `${data.metrics.peak_vram_gb} GB` : "-"
        );

        setPipelineStep("step4", "Image rendered", true);
        setPipelineStep("step5", "Sent to OBS Overlay", true);
    } catch (error) {
        setText("analysis", error.message);
        setPipelineStep("step5", "Error");
        setPreviewError(error.message);
    }
}

// === MEDIA RECORDER LOGIC ===
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
const micBtn = document.getElementById("micBtn");
const micStatus = document.getElementById("micStatus");

async function setupAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        // Try webm first, fallback to standard
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
    
    micBtn.style.background = "#ef4444"; // red
    micBtn.textContent = "🎙️ Recording... Release to send";
    micStatus.textContent = "Listening...";
}

function stopRecording() {
    if (!mediaRecorder || !isRecording) return;
    isRecording = false;
    mediaRecorder.stop();
    
    micBtn.style.background = "#3b82f6"; // blue
    micBtn.textContent = "🎤 Hold Space or Click to Speak";
    micStatus.textContent = "Processing...";
}

// Request permissions immediately
setupAudio();

// Mouse events
micBtn.addEventListener("mousedown", startRecording);
micBtn.addEventListener("mouseup", stopRecording);
micBtn.addEventListener("mouseleave", stopRecording); // if drag out

// Keyboard events (Spacebar)
window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && document.activeElement !== document.getElementById('style')) {
        e.preventDefault();
        startRecording();
    }
});
window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
        stopRecording();
    }
});

updateClock();
setInterval(updateClock, 1000);
resetPipeline();

document.getElementById("clearPreview").addEventListener("click", () => {
    resetPreview();
    // Also tell backend to clear OBS overlays
    fetch(`${BACKEND_BASE_URL}/clear-props`, { method: "POST" }).catch(console.error);
});

// === CUSTOM PROP UPLOAD LOGIC ===
const propUploadBtn = document.getElementById("propUploadBtn");
const propUploadInput = document.getElementById("propUploadInput");
const uploadAnchorSelect = document.getElementById("uploadAnchor");

if (propUploadBtn && propUploadInput) {
    propUploadBtn.addEventListener("click", () => {
        propUploadInput.click();
    });

    propUploadInput.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const anchorType = uploadAnchorSelect.value;
        
        // Reset UI somewhat
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
            propUploadInput.value = ""; // Reset file input
        }
    });
}

const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");

if (sidebar && sidebarToggle) {
    sidebarToggle.addEventListener("click", function () {
        sidebar.classList.toggle("collapsed");
    });
}
