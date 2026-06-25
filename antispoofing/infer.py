r"""
infer.py (antispoofing)
-----------------------
Inference wrapper for anti-spoofing classifier.

Usage:
    from infer import AntiSpoofClassifier
    asf = AntiSpoofClassifier("checkpoints/antispoof/best.pt", device="cuda")
    score = asf.predict(face_bgr)   # 0..1, higher = more "live"
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str = "mobilenetv3") -> nn.Module:
    name = name.lower()
    if name == "mobilenetv3":
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
        return m
    if name == "efficientnetb0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
        return m
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    raise ValueError(name)


class AntiSpoofClassifier:
    def __init__(self, ckpt: str, device: str = "cpu", img_size: int = 224):
        self.device = torch.device(device)
        self.img_size = img_size
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model = build_model(ck.get("backbone", "mobilenetv3")).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        H, W = img_bgr.shape[:2]
        scale = self.img_size / max(H, W)
        nh, nw = int(H * scale), int(W * scale)
        img = cv2.resize(img_bgr, (nw, nh))
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = img
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray) -> float:
        x = self._preprocess(img_bgr)
        logit = self.model(x).squeeze().item()
        return float(1 / (1 + np.exp(-logit)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    asf = AntiSpoofClassifier(args.ckpt, device=args.device)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    s = asf.predict(img)
    print(f"Live score = {s:.4f}  ->  {'LIVE' if s > 0.5 else 'SPOOF'}")


if __name__ == "__main__":
    main()
