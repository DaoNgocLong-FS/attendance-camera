"""
Debug script: Show actual similarity scores for faces in camera
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "recognition"))
sys.path.insert(0, str(ROOT / "pipeline"))

from attendance import AttendancePipeline

# Initialize pipeline
pipeline = AttendancePipeline(
    detector_weights="detection_model/best.pt",
    recognition_ckpt="checkpoints/recognition/last.pt",
    antispoof_ckpt=None,
    db_path="attendance.db",
    device="cuda",
    sim_threshold=0.30,  # Low threshold to see ALL scores
)

print(f"Loaded {len(pipeline.gallery_ids)} employees with {pipeline.gallery_embs.shape[0]} total embeddings\n")
for i, (emp_id, code, name) in enumerate(zip(pipeline.gallery_ids, pipeline.gallery_codes, pipeline.gallery_names)):
    print(f"  Employee {i}: {code} - {name} ({int(np.sum(pipeline.gallery_owner == i))} embeddings)")

print("\nOpening camera (0)... Press ESC to quit.\n")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and process
    results = pipeline.process_frame(frame, log=False)
    
    for result in results:
        # Draw bbox
        x1, y1, x2, y2, score = result.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Show similarity info
        text = f"Sim: {result.sim_score:.3f} | {result.employee_code or 'UNKNOWN'}"
        color = (0, 255, 0) if result.employee_code else (0, 0, 255)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow("Similarity Debug", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
