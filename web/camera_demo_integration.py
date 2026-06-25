"""
camera_demo_integration.py — MẪU TÍCH HỢP 2 tính năng vào vòng lặp camera
=========================================================================
File này MINH HOẠ cách ghép:
  (1) Xác nhận 3 giây liên tục   -> confirm_tracker.py
  (2) Check-in / Check-out        -> attendance_session.py
vào vòng lặp nhận diện hiện có của bạn (camera_demo.py).

CÁC PHẦN [GIỮ NGUYÊN CODE CỦA BẠN] là phần detection/alignment/recognition
sẵn có. Chỉ cần THÊM các dòng đánh dấu [MỚI].

Đây là khung tham khảo — gửi mình camera_demo.py thật để mình ghép chính xác.
"""
import cv2
import time
import sqlite3
import numpy as np
from datetime import datetime

# [MỚI] import 2 module tính năng
from confirm_tracker import ContinuousConfirmTracker
from attendance_session import init_sessions_table, record_attendance, get_today_status


def run(detector_path, recognition_path, db_path, sim_threshold=0.30,
        cam_index=0, required_seconds=3.0, cooldown_sec=10):
    # ---------- [GIỮ NGUYÊN CODE CỦA BẠN] khởi tạo detector + recognizer ----------
    # from ultralytics import YOLO
    # detector = YOLO(detector_path)
    # from extract_embedding import EmbeddingExtractor
    # embedder = EmbeddingExtractor(recognition_path, device="cuda")
    # enrolled = load_enrolled_embeddings(db_path)   # {code: (name, embedding)}
    # aligner = ...  # MediaPipe alignment cua ban
    # ------------------------------------------------------------------------------

    conn = sqlite3.connect(db_path)
    init_sessions_table(conn)                              # [MỚI] tạo bảng phiên nếu chưa có
    tracker = ContinuousConfirmTracker(required_seconds=required_seconds)  # [MỚI]

    cap = cv2.VideoCapture(cam_index)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---------- [GIỮ NGUYÊN] detect + align + recognize mỗi khuôn mặt ----------
        # faces = detector(frame) -> list hộp bao
        # với mỗi face:
        #     aligned = align(frame, face)
        #     emb = embedder.embed_bgr(aligned)
        #     code, name, sim = match(emb, enrolled)   # cosine lớn nhất
        # Giả lập cho minh hoạ:
        detections = []  # [(code, name, sim, (x1,y1,x2,y2)), ...]
        # ---------------------------------------------------------------------------

        seen_now = set()
        for code, name, sim, box in detections:
            x1, y1, x2, y2 = box

            if sim < sim_threshold:
                # Không nhận ra -> khung CAM
                draw_box(frame, box, (0, 165, 255), "Unknown")
                continue

            seen_now.add(code)

            # [MỚI] cập nhật bộ đếm 3 giây liên tục
            progress, just_confirmed = tracker.update(code)

            if just_confirmed:
                # [MỚI] vừa đủ 3 giây -> ghi chấm công (tự quyết IN hay OUT)
                event, t = record_attendance(conn, code, name, cooldown_sec=cooldown_sec)
                if event == "CHECK_IN":
                    flash_message(frame, f"{name}: DA VAO LAM ({t.strftime('%H:%M')})", (0, 200, 0))
                elif event == "CHECK_OUT":
                    flash_message(frame, f"{name}: DA TAN CA ({t.strftime('%H:%M')})", (200, 100, 0))
                # COOLDOWN -> không làm gì

            # [MỚI] vẽ trạng thái + thanh tiến trình 3 giây
            status = get_today_status(conn, code)       # None / checked_in / checked_out
            if tracker.is_confirmed(code):
                label = {"checked_in": "Da vao lam", "checked_out": "Da tan ca"}.get(status, "Da cham cong")
                color = (0, 200, 0)
            else:
                label = f"Dang xac nhan... {progress*required_seconds:.1f}/{required_seconds:.0f}s"
                color = (0, 255, 255)   # vàng khi đang đếm
            draw_box(frame, box, color, f"{name}  {label}")
            draw_progress_bar(frame, box, progress)     # thanh tiến trình dưới khung

        # [MỚI] dọn timer của mặt đã rời khung (để lần sau quay lại tính phiên mới)
        tracker.cleanup()

        cv2.imshow("Cham cong - Nhan dien khuon mat", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


# ---------- Hàm vẽ phụ trợ (minh hoạ) ----------
def draw_box(frame, box, color, text):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_progress_bar(frame, box, progress):
    """Thanh tiến trình xác nhận 3 giây, vẽ dưới khung mặt."""
    x1, y1, x2, y2 = map(int, box)
    w = x2 - x1
    bar_y = y2 + 8
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + 8), (80, 80, 80), -1)
    cv2.rectangle(frame, (x1, bar_y), (x1 + int(w * progress), bar_y + 8), (0, 255, 0), -1)

def flash_message(frame, text, color):
    """Hiện thông báo lớn giữa màn hình khi chấm công thành công."""
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (int(w*0.1), int(h*0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


if __name__ == "__main__":
    # Ví dụ — sửa đường dẫn cho đúng máy bạn
    run(detector_path="detection_model/best.pt",
        recognition_path="checkpoints/recognition/last.pt",
        db_path="attendance.db",
        sim_threshold=0.30,
        cam_index=0,
        required_seconds=3.0,
        cooldown_sec=10)   # demo dùng 10s; thực tế đặt lớn hơn (vd 300s = 5 phút)
