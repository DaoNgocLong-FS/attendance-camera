# Attendance AI — Final System

Hệ thống chấm công bằng nhận diện khuôn mặt với 3 module + alignment.

## Kiến trúc

```
Camera frame
    |
    v
[1] Face Detection (YOLOv8 - đã train trên WIDER FACE)
    |  bbox
    v
    Expand bbox + crop
    |
    v
[+] MediaPipe Face Mesh (5 landmarks - không cần train)
    |  landmarks
    v
[+] Similarity Transform -> aligned 112x112
    |
    v
[2] Face Recognition (IResNet50 + AdaFace, fine-tuned từ pretrained MS1MV2)
    |  512-d embedding
    v
    Cosine similarity với database
    |
    v
[3] Anti-Spoofing (MobileNetV3, trained from scratch trên CelebA-Spoof)
    |  live score
    v
    Decision: log / unknown / spoof / cooldown
    |
    v
SQLite database (attendance.db)
```

## Cấu trúc project

```
attendance-final/
├── recognition/
│   ├── alignment.py             # MediaPipe + similarity transform
│   ├── iresnet.py               # IResNet50 (compat AdaFace)
│   ├── adaface.py               # AdaFace loss head
│   ├── dataset.py               # CASIA-WebFace loader
│   ├── finetune.py              # Fine-tune script
│   ├── download_pretrained.py   # Download AdaFace weights
│   ├── evaluate.py              # LFW evaluation
│   └── extract_embedding.py     # Inference wrapper
├── antispoofing/
│   ├── dataset.py               # CelebA-Spoof loader
│   ├── train.py                 # Train from scratch
│   └── infer.py                 # Inference wrapper
├── pipeline/
│   ├── database.py              # SQLite (employees + attendance)
│   ├── attendance.py            # Main pipeline (combines 3 modules)
│   ├── camera_demo.py           # Webcam realtime demo
│   └── enroll.py                # Enroll employee via webcam
├── datasets/                    # Dataset folder (download here)
├── pretrained/                  # Pretrained weights folder
├── checkpoints/                 # Train outputs
├── requirements.txt
└── README.md
```

---

## ⚠️ Quan trọng cho người dùng Windows

1. **Tất cả lệnh trong README dùng 1 dòng** — copy-paste chạy được trên cả PowerShell lẫn CMD
2. **PowerShell vs CMD**: Khác nhau ở line continuation (` ` ` cho PowerShell, `^` cho CMD). README này tránh hoàn toàn.
3. **Encoding**: Tất cả file code đã được làm sạch (chỉ ASCII trong code, UTF-8 cho ghi file)
4. **Path Windows**: Tránh space và dấu tiếng Việt trong đường dẫn

---

## Bước 0 — Setup môi trường

```powershell
cd D:\attendance-final
```

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\activate
```

Cài PyTorch CUDA:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Cài deps còn lại:
```powershell
pip install -r requirements.txt
```

**⚠️ Đặc biệt:** MediaPipe phải cài version `0.10.13` đến `0.10.15`. Version mới hơn (>=0.10.18) đã bỏ API `solutions` mà code này dùng. requirements.txt đã pin sẵn.

Verify:
```powershell
python -c "import torch, mediapipe as mp; print('torch CUDA:', torch.cuda.is_available()); print('mp version:', mp.__version__)"
```

---

## Bước 1 — Chuẩn bị Module 1 (Detection) — đã có sẵn

Bạn đã train YOLO ở project trước. Copy file `best.pt` vào project này:
```powershell
copy ..\attendance-detection\runs\detect\face_yolov8n\weights\best.pt .\detection_best.pt
```

(Hoặc giữ ở đường dẫn cũ và truyền `--detector D:\path\to\best.pt`.)

Test detection:
```powershell
python -c "from ultralytics import YOLO; m = YOLO('detection_best.pt'); print('OK', m.task)"
```

---

## Bước 2 — Module 2 (Recognition): Tải pretrained + Fine-tune

### 2.1 Tải AdaFace pretrained

```powershell
python recognition\download_pretrained.py
```

Hoặc tải tay từ https://github.com/mk-minchul/AdaFace#pretrained-models (file `adaface_ir50_ms1mv2.ckpt`, ~250MB) và lưu vào `pretrained\adaface_ir50_ms1mv2.ckpt`.

### 2.2 Tải CASIA-WebFace

Tải bản đã align 112x112:
- Kaggle: https://www.kaggle.com/datasets/ntl0601/casia-webface
- Hoặc InsightFace data zoo

Giải nén vào `datasets\casia_webface_aligned\`:
```
datasets\casia_webface_aligned\
    0\000001.jpg
    0\000002.jpg
    1\...
    ...
    (10572 folders, ~494K images)
```

### 2.3 Fine-tune

**Lệnh fine-tune (1 dòng):**

```powershell
python recognition\finetune.py --data datasets\casia_webface_aligned --pretrained pretrained\adaface_ir50_ms1mv2.ckpt --backbone ir_50_se --batch 64 --epochs 5 --lr 1e-4 --workers 4 --amp --out checkpoints\recognition
```

**Tham số theo VRAM:**

| VRAM | --batch | Train time (5 epochs) |
|---|---|---|
| 8 GB (RTX 3060) | 32-48 | ~3-4h |
| 10-12 GB | 64-96 | ~2-3h |
| 16 GB+ | 128 | ~1.5-2h |

### 2.4 Đánh giá LFW

Tải LFW (xem mục Datasets bên dưới), rồi:

```powershell
python recognition\evaluate.py --ckpt checkpoints\recognition\last.pt --root datasets\lfw_aligned_112 --pairs datasets\lfw_pairs.txt
```

Kỳ vọng output:
```
Accuracy 10-fold: 99.50 +/- 0.30%
Best threshold:   0.27
AUC:              0.9995
EER:              0.50%
```

---

## Bước 3 — Module 3 (Anti-Spoofing): Train từ đầu

### 3.1 Tải CelebA-Spoof

CelebA-Spoof full ~80GB, nhưng subset ~20-30GB là đủ cho đồ án.

Tải từ: https://github.com/ZhangYuanhan-AI/CelebA-Spoof

Giải nén vào `datasets\celeba_spoof\` theo cấu trúc:
```
datasets\celeba_spoof\
    Data\
        train\<id>\live\*.jpg
        train\<id>\spoof\*.jpg
        test\<id>\live\*.jpg
        test\<id>\spoof\*.jpg
```

### 3.2 Train

```powershell
python antispoofing\train.py --data datasets\celeba_spoof --backbone mobilenetv3 --batch 64 --epochs 15 --lr 1e-3 --workers 4 --amp --out checkpoints\antispoof
```

Train time: ~5-8h trên RTX 3060.

Output: `checkpoints\antispoof\best.pt`

---

## Bước 4 — Chạy hệ thống chấm công

### 4.1 Enroll nhân viên (đăng ký người dùng)

Đứng trước camera, chạy:

```powershell
python pipeline\enroll.py --detector detection_best.pt --recognition checkpoints\recognition\last.pt --antispoof checkpoints\antispoof\best.pt --db attendance.db --code EMP001 --name "Nguyen Van A" --num-photos 5
```

- Press SPACE để chụp 1 ảnh, hoặc đợi auto-capture
- Mặt phải nằm trong khung xanh
- Chụp 5 ảnh: nhìn thẳng, nghiêng trái, nghiêng phải, cúi nhẹ, ngẩng nhẹ
- Lặp lại lệnh cho từng nhân viên (đổi `--code` và `--name`)

### 4.2 Chạy demo realtime

```powershell
python pipeline\camera_demo.py --detector detection_best.pt --recognition checkpoints\recognition\last.pt --antispoof checkpoints\antispoof\best.pt --db attendance.db --device cuda --sim 0.30 --live 0.70
```

Phím:
- `q`: thoát
- `s`: bật/tắt logging
- `l`: list các record chấm công hôm nay

**Khung màu:**
- 🟢 Xanh lá: đã chấm công thành công
- 🟠 Vàng cam: unknown (không match nhân viên nào)
- 🔴 Đỏ: phát hiện spoof
- 🟡 Vàng nhạt: đã chấm công gần đây (cooldown 5 phút)

---

## Datasets — Links và cấu trúc

### CASIA-WebFace aligned (Module 2 fine-tune)
- Kaggle mirror: https://www.kaggle.com/datasets/ntl0601/casia-webface
- Or InsightFace: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_

### LFW aligned (Module 2 evaluation)
- LFW pairs.txt: http://vis-www.cs.umass.edu/lfw/pairs.txt
- LFW aligned: tìm "LFW 112x112 aligned" trên Kaggle/HuggingFace

### CelebA-Spoof (Module 3 training)
- Official: https://github.com/ZhangYuanhan-AI/CelebA-Spoof

---

## Threshold calibration

Sau khi train xong, threshold trong `camera_demo.py` mặc định:
- `--sim 0.30` — similarity threshold (cosine)
- `--live 0.70` — anti-spoof live threshold

**Để calibrate cho môi trường của bạn:**
1. Enroll 2-3 người, chạy demo
2. Note `sim_score` khi nhận đúng và khi unknown
3. Note `live_score` khi mặt thật và khi spoof (giơ ảnh in ra)
4. Điều chỉnh threshold ở giữa 2 phân phối

Sim_score điển hình:
- Match thật: 0.40 - 0.70
- Mismatch: 0.05 - 0.25
- Threshold tốt: 0.30

Live_score điển hình:
- Mặt thật: 0.85 - 0.99
- Ảnh in giơ ra: 0.10 - 0.40
- Threshold tốt: 0.70

---

## Lý lẽ kỹ thuật cho báo cáo

Cách triển khai này phản ánh practice industry hiện đại:

1. **Module 1 (Detection)**: Train from scratch trên WIDER FACE — dataset đủ lớn, domain phổ quát.

2. **Module 2 (Recognition)**: Fine-tune từ pretrained MS1MV2 (5.8M ảnh, 85K identities) thay vì train from scratch trên CASIA (10K identities). Lý do: face recognition yêu cầu lượng identity rất lớn để học discriminative features. Fine-tune cho phép tận dụng knowledge từ dataset lớn hơn 17x và adapt cho domain hẹp hơn. Đây là best practice trong InsightFace, AdaFace papers.

3. **MediaPipe alignment**: Bước critical mà nhiều người bỏ sót. Face recognition cần input đã align để hoạt động đúng (vì train trên ảnh đã align). Không align → accuracy thực tế thấp hơn benchmark 5-15%.

4. **Module 3 (Anti-Spoofing)**: Train from scratch vì domain shift nghiêm trọng giữa các camera/môi trường/kiểu tấn công. Pretrained anti-spoofing thường fail khi deploy ở môi trường mới. Custom training với CelebA-Spoof là tiếp cận đúng đắn.

---

## Troubleshooting

| Lỗi | Fix |
|---|---|
| `mediapipe has no attribute 'solutions'` | `pip install "mediapipe<=0.10.15"` |
| `torch.cuda.is_available() = False` | Reinstall PyTorch CUDA build |
| `CUDA out of memory` (recognition) | Giảm `--batch` xuống 32 hoặc 16 |
| `CUDA out of memory` (antispoof) | Giảm `--batch` xuống 32 |
| `BrokenPipeError` Windows | Đổi `--workers 4` → `--workers 0` |
| Recognition LFW thấp | Kiểm tra alignment có hoạt động không (mở 1 ảnh aligned bằng cv2.imwrite) |
| Anti-spoof bị reject mặt thật | Threshold `--live` quá cao, giảm xuống 0.5 |
| Recognition không nhận ra ai | Cần enroll thêm ảnh / threshold `--sim` quá cao |

---

## Lộ trình 1-2 tuần

**Tuần 1:**
- Ngày 1: Setup env, copy YOLO weights, tải pretrained AdaFace, tải CASIA
- Ngày 2-3: Fine-tune Module 2 (2-3h), eval LFW
- Ngày 4: Tải CelebA-Spoof, bắt đầu train Module 3
- Ngày 5: Train Module 3 hoàn tất (5-8h)

**Tuần 2:**
- Ngày 6-7: Test alignment + integration pipeline + enroll 3-5 người
- Ngày 8-9: Threshold calibration + edge cases
- Ngày 10-12: Viết báo cáo + ablation (with/without alignment)
- Ngày 13-14: Quay video demo + slide bảo vệ

---

## Báo cáo - các điểm phân tích chính

1. **Comparative analysis**: From-scratch vs Fine-tune (số liệu cụ thể)
2. **Ablation study**: With alignment vs Without alignment (cho thấy tầm quan trọng)
3. **Threshold calibration**: Phân phối sim/live score trên dữ liệu thực
4. **Anti-spoofing analysis**: APCER, BPCER, ACER metrics
5. **System engineering**: Database design, cooldown logic, enrollment flow
6. **Edge cases**: Đeo khẩu trang, kính, ánh sáng kém, người di chuyển nhanh
7. **Future work**: Joint multi-task model, RetinaFace replacement, TensorRT export
