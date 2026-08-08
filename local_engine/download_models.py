"""Ghost Frame — Local Engine Model Weight Downloader

Run this script after installing dependencies to download all required
model weights for the local vision engine.

Usage:
    python download_models.py
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URLS = {
    "rvm_resnet50.pth": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth",
        "Robust Video Matting (ResNet50) — ~108 MB"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "MediaPipe Hand Landmarker — ~7.8 MB"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "MediaPipe Pose Landmarker (Full) — ~9.4 MB"
    ),
}


def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  ✅ Already exists: {dest.name}")
        return True

    print(f"  ⬇️  Downloading {description}...")
    print(f"     URL: {url}")
    print(f"     Destination: {dest}")

    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 // total_size)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r     Progress: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
        print()  # newline after progress
        print(f"  ✅ Downloaded: {dest.name}")
        return True
    except Exception as e:
        print(f"\n  ❌ Failed to download {dest.name}: {e}")
        if dest.exists():
            dest.unlink()  # Remove partial download
        return False


def main():
    print("\n" + "=" * 60)
    print("  Ghost Frame — Local Engine Model Downloader")
    print("=" * 60 + "\n")
    print(f"Download directory: {MODELS_DIR}\n")

    success_count = 0
    total = len(MODEL_URLS)

    for filename, (url, description) in MODEL_URLS.items():
        dest = MODELS_DIR / filename
        if download_file(url, dest, description):
            success_count += 1
        print()

    print("=" * 60)
    print(f"  Results: {success_count}/{total} models ready.")
    if success_count == total:
        print("  🎉 All models downloaded! Local engine is ready.")
    else:
        print("  ⚠️  Some downloads failed. Re-run this script to retry.")
    print("=" * 60 + "\n")

    return 0 if success_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
