r"""
dataset.py (antispoofing)
-------------------------
CelebA-Spoof dataset loader.

Expected structure:
    root/
        Data/
            train/<id>/live/*.jpg     (or just <id>/live/*.jpg)
            train/<id>/spoof/*.jpg
            test/<id>/live/*.jpg
            test/<id>/spoof/*.jpg

Labels:
    1 = live (real face)
    0 = spoof (printed, replayed, mask, etc.)
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_bbox_file(bb_path: Path):
    """CelebA-Spoof bounding box file: x1 y1 x2 y2 score (optional)."""
    if not bb_path.exists():
        return None
    try:
        parts = bb_path.read_text().strip().split()
        if len(parts) < 4:
            return None
        return tuple(int(float(v)) for v in parts[:4])
    except Exception:
        return None


class CelebASpoofDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 224,
        is_train: bool = True,
        use_bbox: bool = True,
        exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        # Try common structure
        base = Path(root) / "Data" / split
        if not base.exists():
            base = Path(root) / split
        if not base.exists():
            raise FileNotFoundError(f"Cannot find {split} split in {root}")

        self.base = base
        self.img_size = img_size
        self.is_train = is_train
        self.use_bbox = use_bbox

        self.samples: List[Tuple[Path, int]] = []
        for img_path in base.rglob("*"):
            if img_path.suffix.lower() not in exts:
                continue
            parent_name = img_path.parent.name.lower()
            if parent_name == "live":
                self.samples.append((img_path, 1))
            elif parent_name == "spoof":
                self.samples.append((img_path, 0))

        if not self.samples:
            raise RuntimeError(f"No live/spoof images found in {base}")

        live = sum(1 for _, l in self.samples if l == 1)
        spoof = len(self.samples) - live
        print(f"[CelebASpoof {split}] live={live}  spoof={spoof}  total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _augment(self, img_rgb: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            img_rgb = np.ascontiguousarray(img_rgb[:, ::-1])
        if np.random.rand() < 0.4:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            img_rgb = np.clip(alpha * img_rgb.astype(np.float32) + beta,
                              0, 255).astype(np.uint8)
        if np.random.rand() < 0.2:
            # Simulate JPEG compression (camera-like)
            quality = np.random.randint(60, 100)
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
            decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return img_rgb

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            if self.use_bbox:
                bb = read_bbox_file(path.parent / (path.stem + "_BB.txt"))
                if bb is not None:
                    x1, y1, x2, y2 = bb
                    H, W = img.shape[:2]
                    bw, bh = x2 - x1, y2 - y1
                    pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
                    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
                    x2 = min(W, x2 + pad_x); y2 = min(H, y2 + pad_y)
                    if x2 > x1 and y2 > y1:
                        img = img[y1:y2, x1:x2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))
        if self.is_train:
            img = self._augment(img)

        # ImageNet normalization (since we use pretrained MobileNetV3)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_f = (img.astype(np.float32) / 255.0 - mean) / std
        return torch.from_numpy(img_f.transpose(2, 0, 1).copy()).float(), label
