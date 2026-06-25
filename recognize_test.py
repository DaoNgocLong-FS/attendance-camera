r"""
recognize_test.py
-----------------
PHASE 2 TEST: Test recognition + detection + alignment on camera stream.

Workflow:
    1. Run script -> camera opens
    2. Stand in front of camera -> press 'r' to REGISTER your face as reference
    3. The system computes embedding from your face and stores it
    4. Now any face detected will show cosine similarity vs your reference
    5. Same person -> high sim (0.5-0.8), Different -> low (0.05-0.25)

Usage:
    python recognize_test.py ^
        --detector detection_model\best.pt ^
        --recognition checkpoints\recognition\test_copy.pt ^
        --cam 1

(Use --cam 1 for iPhone via iVCam, --cam 0 for laptop webcam)

Keys:
    q : quit
    r : register CURRENT detected face as reference
    c : clear reference
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "recognition"))

from ultralytics import YOLO
from alignment import FaceAligner
from extract_embedding import EmbeddingExtractor


def expand_crop(frame, bbox, margin=0.3):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(W, x2 + mx); y2 = min(H, y2 + mx)
    return frame[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--recognition", required=True,
                    help="Recognition checkpoint (use a COPY of last.pt, not last.pt directly)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--rtsp", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--sim-threshold", type=float, default=0.30,
                    help="Cosine threshold to consider as same person")
    args = ap.parse_args()

    # Load detector
    print(f"Loading detector: {args.detector}")
    detector = YOLO(args.detector)

    # Load recognition (the copy of training checkpoint)
    print(f"Loading recognition: {args.recognition}")
    try:
        embedder = EmbeddingExtractor(args.recognition, device=args.device)
    except Exception as e:
        print(f"[ERROR] Cannot load recognition: {e}")
        print("Hint: did you copy last.pt? (don't read it directly while training)")
        return

    # Aligner
    aligner = FaceAligner(output_size=112)

    # Open source
    if args.rtsp:
        cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        print("[FAILED] Cannot open camera")
        return

    print("\nControls: q=quit, r=register reference face, c=clear reference")
    print(f"Sim threshold: {args.sim_threshold} (above = same person)")

    reference_embedding = None  # will store the L2-normalized embedding
    reference_aligned = None     # for showing the reference face in corner
    fps = 0.0
    t0 = time.time()
    frame_count = 0
    last_register_msg_until = 0.0
    register_msg = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - t0)
            t0 = time.time()

        # Detect
        results = detector.predict(source=frame, imgsz=args.imgsz,
                                    conf=args.conf, device=args.device,
                                    verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                score = float(b.conf[0].cpu().numpy())
                boxes.append((int(x1), int(y1), int(x2), int(y2), score))

        # For each detected face: align + embed + compare
        best_current_emb = None  # for 'r' to register
        best_current_aligned = None
        best_area = 0

        for (x1, y1, x2, y2, det_score) in boxes:
            crop = expand_crop(frame, (x1, y1, x2, y2), margin=0.3)
            if crop.size == 0:
                continue
            aligned = aligner.align(crop)
            if aligned is None:
                aligned = cv2.resize(crop, (112, 112))

            # Extract embedding
            emb = embedder.embed_bgr(aligned)

            # Track largest face (for register)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_current_emb = emb
                best_current_aligned = aligned

            # Compare with reference
            if reference_embedding is not None:
                sim = float(np.dot(emb, reference_embedding))
                is_match = sim >= args.sim_threshold
                color = (0, 255, 0) if is_match else (0, 165, 255)
                label = f"MATCH sim={sim:.2f}" if is_match else f"diff sim={sim:.2f}"
            else:
                color = (200, 200, 200)
                label = "no reference (press 'r')"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Show reference face in top-right corner
        if reference_aligned is not None:
            x_off = frame.shape[1] - 120
            y_off = 10
            if y_off + 112 <= frame.shape[0]:
                frame[y_off:y_off + 112, x_off:x_off + 112] = reference_aligned
                cv2.rectangle(frame, (x_off, y_off),
                              (x_off + 112, y_off + 112), (255, 255, 0), 2)
                cv2.putText(frame, "REF", (x_off + 5, y_off + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # HUD
        hud = f"Faces: {len(boxes)}  FPS: {fps:.1f}  Threshold: {args.sim_threshold}"
        cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 2)

        # Register message
        if time.time() < last_register_msg_until and register_msg:
            cv2.putText(frame, register_msg, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Recognition Test", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            if best_current_emb is not None:
                reference_embedding = best_current_emb
                reference_aligned = best_current_aligned.copy()
                register_msg = "REFERENCE REGISTERED!"
                last_register_msg_until = time.time() + 2.0
                print(f"Reference registered. Show your face vs others to compare.")
            else:
                print("No face detected to register.")
        elif k == ord("c"):
            reference_embedding = None
            reference_aligned = None
            print("Reference cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
