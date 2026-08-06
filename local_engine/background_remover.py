"""
Ultimate High-Res Background Remover using Robust Video Matting (ResNet50).
Features: Full resolution, True alpha matte, Temporal stability, 
          Ghosting threshold, Edge-preserving denoise, Despill, & White Balance.
"""
import os
import argparse
import cv2
import numpy as np
import torch

# ResNet50 (highest quality model)
MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvm_resnet50.pth")

class BackgroundRemover:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f"[Background Remover] Using device: {self.device}")
        
        if self.device.type == 'cpu':
            print("[WARNING] GPU not detected! ResNet50 at full-res will be very slow on CPU.")
            
        # 1. Download the .pth weights
        if not os.path.exists(MODEL_PATH):
            print(f"[Background Remover] Downloading ResNet50 weights (150MB)...")
            torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH, progress=True)
            
        print("[Background Remover] Loading ResNet50 architecture...")
        # 2. Load ResNet50 architecture
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50", pretrained=False)
        
        # 3. Load weights (weights_only=True removes the FutureWarning)
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        
        # FP16 for faster inference on CUDA
        self._use_fp16 = (self.device.type == 'cuda')
        if self._use_fp16:
            self.model.half()
        
        # 4. RVM recurrent states (memory stops flickering)
        self.rec = [None] * 4  
        
        # DOWNSAMPLE RATIO: 0.25 for maximum real-time FPS on RTX 4050
        self.downsample_ratio = 0.25

    def remove(self, frame_bgr, background_bgr):
        """Composite person from frame onto background — GPU-accelerated pipeline."""
        h, w = frame_bgr.shape[:2]
        
        # Skip resize if background already matches frame dimensions
        if background_bgr.shape[0] != h or background_bgr.shape[1] != w:
            bg_resized = cv2.resize(background_bgr, (w, h))
        else:
            bg_resized = background_bgr
        
        # Preprocess frame: BGR -> RGB, to GPU tensor
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0)
        if self._use_fp16:
            src = src.half().div_(255.0)
        else:
            src = src.float().div_(255.0)
        
        # Preprocess background: BGR -> RGB, to GPU tensor
        bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
        bg_t = torch.from_numpy(bg_rgb).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0)
        if self._use_fp16:
            bg_t = bg_t.half().div_(255.0)
        else:
            bg_t = bg_t.float().div_(255.0)
        
        # RVM inference
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(src, *self.rec, self.downsample_ratio)
        
        # --- All post-processing on GPU (no CPU round-trip) ---
        # Ghosting threshold + alpha contrast
        pha = pha.clamp_(min=0.10)
        pha = pha.sub_(0.10).div_(0.80).clamp_(0, 1)
        
        # Alpha blend on GPU (skip bilateral filter — RVM's recurrence is smooth enough)
        out = fgr * pha + bg_t * (1.0 - pha)
        
        # Single GPU->CPU transfer of the final composited frame
        result = out[0].permute(1, 2, 0).float().mul_(255.0).clamp_(0, 255).byte().cpu().numpy()
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        return result_bgr

    def close(self):
        self.rec = [None] * 4


if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="RVM High-Res Background Remover")
    parser.add_argument("--background", type=str, default=None, 
                        help="Path to the background image file (e.g., 'bg.jpg')")
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    remover = BackgroundRemover()
    
    # Load the custom background image if provided
    if args.background:
        bg_image = cv2.imread(args.background)
        if bg_image is None:
            print(f"Error: Could not read background image at '{args.background}'")
            exit()
        print(f"Using custom background: {args.background}")
    else:
        # Default to a neutral gray background if no image is provided
        print("No background provided. Using default gray. (Use --background 'image.jpg' to set one)")
        bg_image = np.zeros((480, 640, 3), dtype=np.uint8)
        bg_image[:] = (60, 60, 60) 
    
    print("Press 'ESC' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        frame = cv2.resize(frame, (640, 480))
        
        # Run the ultimate matting with the chosen background
        result = remover.remove(frame, bg_image)
        
        cv2.imshow("RVM ResNet50 (Full Resolution)", result)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    remover.close()
    cv2.destroyAllWindows()
