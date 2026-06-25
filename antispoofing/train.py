r"""
train.py (antispoofing)
-----------------------
Train MobileNetV3-small for live/spoof binary classification on CelebA-Spoof.

Strategy:
    - MobileNetV3-small with ImageNet pretrained
    - Replace classifier head with 1 output logit (sigmoid)
    - AdamW + cosine LR
    - 15 epochs (binary task converges fast)
    - Heavy augmentation (color, blur, JPEG compression) to be robust

Metrics reported:
    - Accuracy, Precision, Recall, F1
    - APCER (Attack Presentation Classification Error Rate)
    - BPCER (Bona-fide Presentation Classification Error Rate)
    - ACER (Average Classification Error Rate) = (APCER + BPCER) / 2

Usage on Windows:
    python antispoofing\train.py ^
        --data datasets\celeba_spoof ^
        --epochs 15 ^
        --batch 64 ^
        --lr 1e-3 ^
        --workers 4 ^
        --amp ^
        --out checkpoints\antispoof
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from dataset import CelebASpoofDataset


def build_model(name: str = "mobilenetv3") -> nn.Module:
    name = name.lower()
    if name == "mobilenetv3":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
        return m
    if name == "efficientnetb0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 1)
        return m
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = tn = fn = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        logits = model(imgs).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        lbl = labels.long()
        tp += int(((preds == 1) & (lbl == 1)).sum())
        tn += int(((preds == 0) & (lbl == 0)).sum())
        fp += int(((preds == 1) & (lbl == 0)).sum())
        fn += int(((preds == 0) & (lbl == 1)).sum())

    eps = 1e-9
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    apcer = fp / (fp + tn + eps)  # spoof predicted as live
    bpcer = fn / (fn + tp + eps)  # live rejected
    acer = (apcer + bpcer) / 2
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "APCER": apcer, "BPCER": bpcer, "ACER": acer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", default="mobilenetv3",
                    choices=["mobilenetv3", "efficientnetb0", "resnet18"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/antispoof")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CelebASpoofDataset(args.data, "train",
                                   img_size=args.img_size, is_train=True)
    val_ds = CelebASpoofDataset(args.data, "test",
                                 img_size=args.img_size, is_train=False)

    train_loader = DataLoader(train_ds, args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    best_acer = 1.0
    log_path = out_dir / "log.txt"
    log_file = open(log_path, "a", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(imgs).squeeze(1)
                loss = loss_fn(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            n += imgs.size(0)

        scheduler.step()
        train_loss = loss_sum / max(1, n)
        elapsed = time.time() - t0

        metrics = evaluate(model, val_loader, device)
        msg = (f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
               f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}  "
               f"APCER={metrics['APCER']:.4f}  BPCER={metrics['BPCER']:.4f}  "
               f"ACER={metrics['ACER']:.4f}  ({elapsed:.1f}s)")
        print(msg)
        log_file.write(msg + "\n"); log_file.flush()

        torch.save({
            "model_state": model.state_dict(),
            "backbone": args.backbone,
            "epoch": epoch,
            "metrics": metrics,
        }, out_dir / "last.pt")

        if metrics["ACER"] < best_acer:
            best_acer = metrics["ACER"]
            torch.save({
                "model_state": model.state_dict(),
                "backbone": args.backbone,
                "epoch": epoch,
                "metrics": metrics,
            }, out_dir / "best.pt")
            print(f"  -> best.pt updated (ACER={best_acer:.4f})")

    log_file.close()


if __name__ == "__main__":
    main()
