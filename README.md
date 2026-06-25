# Hệ thống Chấm công bằng Nhận diện Khuôn mặt

> **Attendance AI** — Chấm công tự động qua camera, kết hợp **phát hiện khuôn mặt**,
> **căn chỉnh (alignment)**, **nhận diện danh tính**, **chống giả mạo (anti-spoofing)**,
> lưu vào cơ sở dữ liệu và hiển thị **dashboard web check-in / check-out**.

Dự án gồm 3 mô-đun học sâu độc lập + 1 bước căn chỉnh, được ghép thành một pipeline
thời gian thực, kèm theo công cụ đăng ký nhân viên và một web app quản lý.

---

## Mục lục

1. [Tính năng nổi bật](#1-tính-năng-nổi-bật)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Tải model & dữ liệu](#5-tải-model--dữ-liệu)
6. [Giải thích 4 thành phần lõi](#6-giải-thích-4-thành-phần-lõi)
7. [Huấn luyện model](#7-huấn-luyện-model)
8. [Chạy hệ thống](#8-chạy-hệ-thống)
9. [Web Dashboard](#9-web-dashboard)
10. [Cơ sở dữ liệu](#10-cơ-sở-dữ-liệu)
11. [Ngưỡng & hiệu chỉnh](#11-ngưỡng--hiệu-chỉnh)
12. [Lý lẽ kỹ thuật (cho báo cáo)](#12-lý-lẽ-kỹ-thuật-cho-báo-cáo)
13. [Khắc phục sự cố](#13-khắc-phục-sự-cố)
14. [Công nghệ sử dụng](#14-công-nghệ-sử-dụng)

---

## 1. Tính năng nổi bật

- 🎯 **Nhận diện khuôn mặt thời gian thực** qua webcam / camera IP (RTSP) / iPhone (iVCam).
- 🧭 **Căn chỉnh khuôn mặt (alignment)** bằng MediaPipe — bước then chốt giúp nhận diện chính xác.
- 🛡️ **Chống giả mạo (anti-spoofing)** — phát hiện ảnh in / màn hình phát lại (tùy chọn bật/tắt).
- 🗄️ **Lưu trữ khoa học** trong SQLite: hồ sơ nhân viên (embedding) + log chấm công.
- ⏱️ **Dashboard web check-in / check-out**: trục thời gian trực quan cho biết **ai vào/ra lúc nào trong ngày**.
- 🔌 **Thiết kế mô-đun**: 3 model tách rời, có thể train / thay thế độc lập.

---

## 2. Kiến trúc tổng thể

```
                         ┌──────────────────────────┐
   Camera frame  ──────► │  [1] FACE DETECTION       │  detection_model/best.pt
   (webcam/RTSP)         │      YOLOv8 (WIDER FACE)  │  → bounding box
                         └────────────┬─────────────┘
                                      │ crop + mở rộng 30%
                                      ▼
                         ┌──────────────────────────┐
                         │  [+] FACE ALIGNMENT       │  recognition/alignment.py
                         │      MediaPipe 5 điểm mốc │  → ảnh 112×112 chuẩn ArcFace
                         │      + similarity transform
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  [2] FACE RECOGNITION     │  checkpoints/recognition/last.pt
                         │      IResNet-50 + AdaFace │  → embedding 512 chiều
                         │      (fine-tune MS1MV2)   │
                         └────────────┬─────────────┘
                                      │ cosine similarity với gallery (DB)
                                      ▼
                         ┌──────────────────────────┐
                         │  [3] ANTI-SPOOFING (tùy chọn)  checkpoints/antispoof/best.pt
                         │      MobileNetV3 (CelebA-Spoof)│ → live score 0..1
                         └────────────┬─────────────┘
                                      │ quyết định: logged / unknown / spoof / cooldown
                                      ▼
                         ┌──────────────────────────┐
                         │  SQLite  attendance.db    │  pipeline/database.py
                         │  employees + attendance   │
                         └────────────┬─────────────┘
                                      │ suy ra phiên check-in/out
                                      ▼
                         ┌──────────────────────────┐
                         │  WEB DASHBOARD (Flask)    │  web/app.py + templates/index.html
                         │  Tổng quan · Timeline ·   │
                         │  Lịch sử · Nhân viên      │
                         └──────────────────────────┘
```

---

## 3. Cấu trúc thư mục

```
attendance-camera/
├── detection_model/
│   ├── best.pt                     # YOLOv8 face detector (PyTorch)
│   └── best.onnx                   # Bản export ONNX để deploy
│
├── recognition/                    # MÔ-ĐUN 2: Nhận diện + Căn chỉnh
│   ├── alignment.py                # MediaPipe Face Mesh → 5 landmark → 112×112
│   ├── iresnet.py                  # Backbone IResNet (khớp kiến trúc AdaFace chính thức)
│   ├── adaface.py                  # AdaFace head (quality-adaptive margin, CVPR 2022)
│   ├── dataset.py                  # Loader CASIA-WebFace (folder-per-identity)
│   ├── finetune.py                 # Fine-tune từ pretrained MS1MV2
│   ├── extract_embedding.py        # Wrapper inference: ảnh → embedding 512-d
│   ├── evaluate.py                 # Đánh giá trên LFW (accuracy, AUC, EER)
│   ├── download_pretrained.py      # Tải weights AdaFace pretrained
│   └── diagnose_recognition.py     # Công cụ debug recognition
│
├── antispoofing/                   # MÔ-ĐUN 3: Chống giả mạo
│   ├── dataset.py                  # Loader CelebA-Spoof (live/spoof)
│   ├── train.py                    # Train MobileNetV3 từ đầu (APCER/BPCER/ACER)
│   └── infer.py                    # Wrapper inference: face → live score
│
├── pipeline/                       # GHÉP 3 MÔ-ĐUN
│   ├── attendance.py               # AttendancePipeline — lõi của hệ thống
│   ├── camera_demo.py              # Demo realtime qua webcam
│   ├── enroll.py                   # Đăng ký nhân viên qua webcam
│   ├── database.py                 # SQLite ORM (SQLAlchemy)
│   └── test_rtsp.py                # Test camera IP qua RTSP
│
├── web/                            # WEB DASHBOARD (Flask)
│   ├── app.py                      # Backend: suy ra phiên check-in/out từ log
│   ├── templates/index.html        # Giao diện (tabs + timeline, không cần internet)
│   ├── attendance_session.py       # Logic phiên check-in/out (tham khảo)
│   ├── confirm_tracker.py          # Bộ đếm xác nhận N giây liên tục (tham khảo)
│   ├── camera_demo_integration.py  # Mẫu tích hợp vào vòng lặp camera
│   ├── requirements_web.txt        # Phụ thuộc riêng cho web (Flask)
│   └── README_web.md               # Hướng dẫn riêng cho web
│
├── checkpoints/
│   ├── recognition/last.pt         # ⚠️ Model nhận diện đã fine-tune (~375MB, tải riêng)
│   └── antispoof/best.pt           # Model chống giả mạo (~6MB, có sẵn trong repo)
│
├── check_db.py                     # Xem nhanh nội dung database
├── detect_camera.py                # Test riêng module detection
├── recognize_test.py               # Test pipeline nhận diện trên ảnh tĩnh
├── test_camera.py                  # Test mở webcam
├── test_similarity.py              # So sánh độ giống của 2 khuôn mặt
├── all_charts_recognition_1.ipynb  # Notebook vẽ biểu đồ kết quả recognition
├── train_recognition_colab (1).ipynb # Notebook train recognition trên Colab
├── attendance.db                   # CSDL runtime (sinh ra khi chạy, không commit)
├── requirements.txt
└── README.md
```

---

## 4. Cài đặt môi trường

> Khuyến nghị: Windows + Python 3.10 + GPU NVIDIA (CUDA). Vẫn chạy được CPU nhưng chậm.

```powershell
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\activate

# 2. Cài PyTorch bản CUDA (chỉnh cu121 theo phiên bản CUDA của bạn)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Cài các thư viện còn lại
pip install -r requirements.txt
```

**⚠️ Lưu ý MediaPipe:** phải dùng bản `0.10.13` → `0.10.15` (đã ghim trong `requirements.txt`).
Bản mới hơn bỏ API `mp.solutions` mà `alignment.py` đang dùng.

Kiểm tra nhanh:
```powershell
python -c "import torch, mediapipe as mp; print('CUDA:', torch.cuda.is_available(), '| mediapipe:', mp.__version__)"
```

---

## 5. Tải model & dữ liệu

| Thành phần | Kích thước | Nguồn |
|---|---|---|
| `detection_model/best.pt` | ~6 MB | ✅ Có sẵn trong repo |
| `checkpoints/antispoof/best.pt` | ~6 MB | ✅ Có sẵn trong repo |
| `checkpoints/recognition/last.pt` | **~375 MB** | ⬇️ **Tải riêng** (vượt giới hạn 100MB của GitHub) |
| AdaFace pretrained `adaface_ir50_ms1mv2.ckpt` | ~250 MB | Chỉ cần khi train lại — xem [§7](#7-huấn-luyện-model) |

> **Tải model nhận diện đã fine-tune (`last.pt`):**
> 👉 *(Dán link Google Drive của bạn vào đây)*
> Sau khi tải, đặt file vào: `checkpoints/recognition/last.pt`

**Dataset** (chỉ cần nếu muốn train lại):
- **CASIA-WebFace** (recognition): [Kaggle](https://www.kaggle.com/datasets/ntl0601/casia-webface) — bản align 112×112, ~494K ảnh / 10.572 người.
- **LFW** (đánh giá recognition): [pairs.txt](http://vis-www.cs.umass.edu/lfw/pairs.txt) + bộ ảnh align 112×112.
- **CelebA-Spoof** (anti-spoofing): [GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof).

---

## 6. Giải thích 4 thành phần lõi

### [1] Phát hiện khuôn mặt — `detection_model/best.pt`
- **YOLOv8** huấn luyện trên **WIDER FACE**, trả về bounding box + độ tin cậy.
- Chạy bằng `ultralytics`; có bản `best.onnx` để deploy không cần PyTorch.

### [+] Căn chỉnh — `recognition/alignment.py`
- **MediaPipe Face Mesh** lấy 468 landmark → chọn **5 điểm** (2 mắt, mũi, 2 khóe miệng).
- Tính **similarity transform** đưa về 5 điểm chuẩn ArcFace, warp ra ảnh **112×112**.
- Đây là bước nhiều người bỏ sót: thiếu nó, accuracy thực tế giảm **5–15%** vì model
  recognition được train trên ảnh đã align.

### [2] Nhận diện — `checkpoints/recognition/last.pt`
- Backbone **IResNet-50** (`iresnet.py`) + head **AdaFace** (`adaface.py`, CVPR 2022).
- **Fine-tune** từ pretrained **MS1MV2** trên **CASIA-WebFace** (10.572 danh tính).
- Sinh **embedding 512 chiều** đã L2-normalize; nhận đầu vào **BGR** chuẩn hóa về `[-1, 1]`.
- So khớp danh tính bằng **cosine similarity** với gallery trong DB (phép nhân ma trận).
- `extract_embedding.py` **tự dò kiến trúc** từ trọng số để tránh lỗi lệch `ir_50` / `ir_50_se`.

### [3] Chống giả mạo (tùy chọn) — `checkpoints/antispoof/best.pt`
- **MobileNetV3-small** (có thể đổi sang EfficientNet-B0 / ResNet18) train **từ đầu** trên CelebA-Spoof.
- Phân loại nhị phân **live (1) / spoof (0)**; đầu vào 224×224 chuẩn hóa ImageNet.
- Báo cáo **APCER / BPCER / ACER** — bộ chỉ số chuẩn của bài toán anti-spoofing.
- **Tùy chọn**: nếu không truyền `--antispoof`, pipeline coi mọi khuôn mặt là live (score 1.0).

---

## 7. Huấn luyện model

> Bỏ qua mục này nếu chỉ chạy demo với model đã có.

### 7.1 Recognition (fine-tune)
```powershell
# Tải pretrained AdaFace
python recognition\download_pretrained.py

# Fine-tune trên CASIA-WebFace (1 dòng)
python recognition\finetune.py --data datasets\casia_webface_aligned --pretrained pretrained\adaface_ir50_ms1mv2.ckpt --backbone ir_50_se --batch 64 --epochs 5 --lr 1e-4 --workers 4 --amp --out checkpoints\recognition
```

| VRAM | `--batch` | Thời gian (5 epoch) |
|---|---|---|
| 8 GB (RTX 3060) | 32–48 | ~3–4h |
| 10–12 GB | 64–96 | ~2–3h |
| 16 GB+ | 128 | ~1.5–2h |

Đánh giá trên LFW:
```powershell
python recognition\evaluate.py --ckpt checkpoints\recognition\last.pt --root datasets\lfw_aligned_112 --pairs datasets\lfw_pairs.txt
```
Kỳ vọng: Accuracy ~99.5%, AUC ~0.9995, EER ~0.5%.

### 7.2 Anti-spoofing (train từ đầu)
```powershell
python antispoofing\train.py --data datasets\celeba_spoof --backbone mobilenetv3 --batch 64 --epochs 15 --lr 1e-3 --workers 4 --amp --out checkpoints\antispoof
```
Train ~5–8h trên RTX 3060 → sinh `checkpoints\antispoof\best.pt`.

---

## 8. Chạy hệ thống

### 8.1 Đăng ký nhân viên
Đứng trước camera, chụp 5 ảnh (thẳng, nghiêng trái/phải, cúi/ngẩng nhẹ):
```powershell
python pipeline\enroll.py --detector detection_model\best.pt --recognition checkpoints\recognition\last.pt --db attendance.db --code EMP001 --name "Nguyen Van A" --num-photos 5 --cam 0
```
- `SPACE` = chụp · `ESC` = hủy. Lặp lại với `--code` / `--name` khác cho từng người.

### 8.2 Demo chấm công realtime
```powershell
python pipeline\camera_demo.py --detector detection_model\best.pt --recognition checkpoints\recognition\last.pt --db attendance.db --device cuda --cam 0 --sim 0.30
```
Thêm `--antispoof checkpoints\antispoof\best.pt` để bật chống giả mạo.

**Phím tắt:** `q` thoát · `s` bật/tắt ghi log · `l` liệt kê chấm công hôm nay · `r` xoay ảnh 180°.

**Màu khung:** 🟢 đã chấm công · 🟠 unknown · 🔴 spoof · 🟡 cooldown (vừa chấm gần đây).

---

## 9. Web Dashboard

Web app đọc `attendance.db` và **suy ra phiên làm việc** từ log sự kiện thô:

> Với mỗi **(nhân viên, ngày)**: lần nhận diện **đầu** = *giờ vào*, lần **cuối** = *giờ ra*,
> hiệu số = *số giờ làm*. Cách này dùng được toàn bộ dữ liệu đã có mà **không cần sửa pipeline camera**.

```powershell
cd web
pip install -r requirements_web.txt
python app.py --db ..\attendance.db
```
Mở trình duyệt: **http://localhost:5000**

**4 tab:**
- **📊 Tổng quan** — thẻ số liệu (vào làm / tan ca / đang làm) + biểu đồ phân bố giờ vào & số phiên 14 ngày.
- **⏱️ Dòng thời gian** — *điểm nhấn*: trục giờ 6h–22h, mỗi người một thanh **giờ vào → giờ ra**; chọn ngày bất kỳ.
- **📋 Lịch sử** — bảng phiên gộp theo ngày (giờ vào/ra, số giờ, số lần nhận diện, trạng thái).
- **👥 Nhân viên** — danh sách đăng ký + số ngày đi làm + lần cuối xuất hiện.

> 💡 **Demo hybrid (2 cửa sổ):** chạy `camera_demo.py` ở terminal 1 (ghi DB) và `web/app.py`
> ở terminal 2; bật "Tự làm mới (5s)" trên web để cập nhật trực tiếp. Xem thêm
> [web/README_web.md](web/README_web.md).

---

## 10. Cơ sở dữ liệu

SQLite quản lý bằng SQLAlchemy (`pipeline/database.py`):

**Bảng `employees`** — hồ sơ nhân viên
| Cột | Ý nghĩa |
|---|---|
| `id`, `code`, `name` | Khóa chính, mã NV (duy nhất), họ tên |
| `embeddings_blob` | Nhiều embedding 512-d / người, lưu dạng blob (đã L2-normalize) |
| `embedding_dim`, `num_embeddings` | Số chiều & số embedding |
| `created_at` | Thời điểm đăng ký |

**Bảng `attendance`** — log sự kiện nhận diện
| Cột | Ý nghĩa |
|---|---|
| `id`, `employee_id` | Khóa chính, tham chiếu nhân viên |
| `timestamp` | Thời điểm nhận diện |
| `sim_score`, `live_score` | Độ tương đồng cosine & điểm liveness |

Logic **cooldown** (mặc định 5 phút) tránh ghi trùng khi một người đứng lâu trước camera.

---

## 11. Ngưỡng & hiệu chỉnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--sim` | 0.30 | Ngưỡng cosine để công nhận danh tính |
| `--live` | 0.70 | Ngưỡng liveness để coi là mặt thật |
| `--cooldown` | 5 | Số phút tối thiểu giữa 2 lần chấm công của cùng người |

Phân phối điển hình: khớp đúng `sim` 0.40–0.70, sai 0.05–0.25 → ngưỡng tốt ~0.30.
Mặt thật `live` 0.85–0.99, ảnh in 0.10–0.40 → ngưỡng tốt ~0.70.
Hãy enroll vài người, quan sát điểm số thực tế và đặt ngưỡng vào giữa hai phân phối.

---

## 12. Lý lẽ kỹ thuật (cho báo cáo)

1. **Detection — train from scratch** trên WIDER FACE: dataset đủ lớn, domain phổ quát.
2. **Recognition — fine-tune** từ MS1MV2 (5.8M ảnh / 85K danh tính) thay vì train from scratch
   trên CASIA (10K danh tính). Nhận diện cần lượng danh tính cực lớn để học đặc trưng
   phân biệt; fine-tune tận dụng dataset lớn gấp ~17× — đúng best practice của InsightFace/AdaFace.
3. **Alignment** là bước critical: model recognition được train trên ảnh đã align, nên input
   thực tế cũng phải align để đạt đúng benchmark.
4. **Anti-spoofing — train from scratch** vì domain shift nặng giữa camera / môi trường /
   kiểu tấn công; pretrained anti-spoofing thường fail khi deploy môi trường mới.

**Các điểm phân tích nên đưa vào báo cáo:** so sánh from-scratch vs fine-tune · ablation
có/không alignment · hiệu chỉnh ngưỡng trên dữ liệu thật · APCER/BPCER/ACER · thiết kế DB &
cooldown · edge case (khẩu trang, kính, thiếu sáng) · hướng phát triển (multi-task, RetinaFace, TensorRT).

---

## 13. Khắc phục sự cố

| Lỗi | Cách khắc phục |
|---|---|
| `mediapipe has no attribute 'solutions'` | `pip install "mediapipe<=0.10.15"` |
| `torch.cuda.is_available() = False` | Cài lại PyTorch bản CUDA |
| `CUDA out of memory` | Giảm `--batch` (32 hoặc 16) |
| `BrokenPipeError` (Windows) | Đổi `--workers` về `0` |
| Recognition không nhận ra ai | Enroll thêm ảnh, hoặc giảm `--sim` |
| Anti-spoof từ chối mặt thật | Giảm `--live` xuống ~0.5 |
| Web báo "Chưa tìm thấy database" | Truyền đúng đường dẫn: `python app.py --db ..\attendance.db` |
| Camera lộn ngược (iVCam) | Nhấn `r` trong demo, hoặc thêm `--rotate180` |

---

## 14. Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.10
- **Học sâu:** PyTorch · torchvision · Ultralytics YOLOv8
- **Thị giác máy tính:** OpenCV · MediaPipe
- **Nhận diện:** IResNet-50 · AdaFace (CVPR 2022)
- **Chống giả mạo:** MobileNetV3 / EfficientNet / ResNet
- **CSDL:** SQLite · SQLAlchemy
- **Web:** Flask (backend) · HTML/CSS/JavaScript thuần (frontend, không CDN)
- **Tiện ích:** NumPy · scikit-learn · tqdm · Pillow · gdown

---

*Đồ án Nhận diện Khuôn mặt — Hệ thống Chấm công AI.*
