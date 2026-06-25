r"""
camera_demo.py  (anti-spoofing OPTIONAL)
----------------------------------------
Realtime attendance demo.

Usage WITHOUT anti-spoofing (current stage), iPhone via iVCam:
    python pipeline\camera_demo.py ^
        --detector detection_model\best.pt ^
        --recognition checkpoints\recognition\last.pt ^
        --db attendance.db ^
        --device cuda ^
        --cam 1 ^
        --sim 0.30

Add --antispoof checkpoints\antispoof\best.pt later when trained.

Keys:
    q : quit
    s : toggle logging on/off
    l : list today's attendance records
    r : rotate frame 180° (nếu IVCam bị lộn ngược)
"""

import argparse
import sys
import threading
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.attendance import AttendancePipeline, draw_result


class ThreadedCamera:
    """Đọc camera trên thread riêng — tránh đơ khi pipeline xử lý chậm hơn FPS camera."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while True:
            ok, frame = self.cap.read()
            with self._lock:
                self._ok = ok
                self._frame = frame
            if not ok:
                break

    def read(self):
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--recognition", required=True)
    ap.add_argument("--antispoof", default=None,
                    help="Optional. Skip if not trained yet.")
    ap.add_argument("--db", default="attendance.db")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--rtsp", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sim", type=float, default=0.30)
    ap.add_argument("--live", type=float, default=0.70)
    ap.add_argument("--det-conf", type=float, default=0.4)
    ap.add_argument("--cooldown", type=int, default=5)
    ap.add_argument("--confirm", type=float, default=2.0,
                    help="So giay nhan dien lien tuc truoc khi cham cong (chong ghi trung)")
    ap.add_argument("--rotate180", action="store_true",
                    help="Xoay ảnh 180° (dùng khi IVCam bị lộn ngược)")
    args = ap.parse_args()

    print("Initializing pipeline...")
    pipeline = AttendancePipeline(
        detector_weights=args.detector,
        recognition_ckpt=args.recognition,
        antispoof_ckpt=args.antispoof,
        db_path=args.db,
        device=args.device,
        det_conf=args.det_conf,
        sim_threshold=args.sim,
        live_threshold=args.live,
        cooldown_min=args.cooldown,
        confirm_seconds=args.confirm,
    )
    print(f"Gallery: {len(pipeline.gallery_ids)} employees, "
          f"{int(pipeline.gallery_embs.shape[0])} embeddings")

    if args.rtsp:
        cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = ThreadedCamera(args.cam)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera source")

    rotate = args.rotate180
    logging_on = True
    print("\nControls: q=quit, s=toggle log, l=list today, r=rotate 180°")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        results = pipeline.process_frame(frame, log=logging_on)
        frame = draw_result(frame, results)
        spoof_status = "ON" if pipeline.antispoof else "OFF(not trained)"
        rot_tag = " ROT180" if rotate else ""
        status = f"LOG:{'ON' if logging_on else 'OFF'}  Spoof:{spoof_status}  ppl:{len(pipeline.gallery_ids)}{rot_tag}"
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0) if logging_on else (180, 180, 180), 2)
        cv2.imshow("Attendance Demo", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            logging_on = not logging_on
            print(f"Logging: {'ON' if logging_on else 'OFF'}")
        elif k == ord("r"):
            rotate = not rotate
            print(f"Rotate 180°: {'ON' if rotate else 'OFF'}")
        elif k == ord("l"):
            records = pipeline.db.list_today()
            print(f"\n=== Today's attendance ({len(records)}) ===")
            for r in records:
                print(f"  {r['timestamp'][:19]}  {r['employee_code']}  {r['name']}  sim={r['sim_score']:.3f}")
            print()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
