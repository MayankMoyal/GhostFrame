const BACKEND_BASE_URL = "https://8000-01kwc2kesrrzxr5h45zgh9ztr4.cloudspaces.litng.ai";

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
            headers: {
                "Content-Type": "application/json",
            },
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

        setText("analysis", "Completed");
        setText("sceneType", "Generated image");
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
        setPipelineStep("step5", "Ready", true);
    } catch (error) {
        setText("analysis", error.message);
        setPipelineStep("step5", "Error");
        setPreviewError(error.message);
    } finally {
        generateBtn.textContent = "Generate Image";
        generateBtn.disabled = false;
    }
}

updateClock();
setInterval(updateClock, 1000);
resetPipeline();

document.getElementById("generate").addEventListener("click", generateImage);
document.getElementById("clearPreview").addEventListener("click", resetPreview);

const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");

if (sidebar && sidebarToggle) {
    sidebarToggle.addEventListener("click", function () {
        sidebar.classList.toggle("collapsed");
    });
}
