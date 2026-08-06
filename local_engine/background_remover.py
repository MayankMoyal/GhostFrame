"""Optimized High-Speed Background Remover using Robust Video Matting (MobileNetV3)."""
import os
import argparse
import cv2
import numpy as np
import torch

MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvm_mobilenetv3.pth")

class BackgroundRemover:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Background Remover] Using device: {self.device}")
        
        if not os.path.exists(MODEL_PATH):
            print(f"[Background Remover] Downloading MobileNetV3 weights (40MB)...")
            torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH, progress=True)
            
        print("[Background Remover] Loading MobileNetV3 architecture...")
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", pretrained=False)
        
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        
        self.rec = [None] * 4  
        self.downsample_ratio = 0.5 if self.device.type == 'cuda' else 0.25

    def remove(self, frame_bgr, background_bgr):
        h, w = frame_bgr.shape[:2]
        bg_resized = cv2.resize(background_bgr, (w, h))
        
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src = torch.from_numpy(rgb).to(self.device).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(src, *self.rec, self.downsample_ratio)
        
        alpha = pha[0, 0].cpu().numpy()[..., None]
        fgr_np = fgr[0].permute(1, 2, 0).cpu().numpy()
        fgr_np = (fgr_np * 255.0).astype(np.float32)
        fgr_bgr = cv2.cvtColor(fgr_np, cv2.COLOR_RGB2BGR)
        
        out = (fgr_bgr * alpha + bg_resized.astype(np.float32) * (1.0 - alpha))
        return out.astype(np.uint8)

    def close(self):
        self.rec = [None] * 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=str, default=None)
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    remover = BackgroundRemover()
    
    if args.background:
        bg_image = cv2.imread(args.background)
    else:
        bg_image = np.zeros((480, 640, 3), dtype=np.uint8)
        bg_image[:] = (60, 60, 60) 
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        # FIX: Flip horizontally (1) for standard selfie/mirror view.
        frame = cv2.flip(frame, 1) 
        
        result = remover.remove(frame, bg_image)
        cv2.imshow("Optimized RVM", result)
        if cv2.waitKey(1) & 0xFF == 27: break
            
    cap.release()
    remover.close()
    cv2.destroyAllWindows()