import cv2
import sys

print("Testing camera...")
for cam_idx in [0, 1, -1]:
    print(f"\n--- Testing camera {cam_idx} ---")
    cap = cv2.VideoCapture(cam_idx)
    if cap.isOpened():
        print(f"✓ Camera {cam_idx} OPENED")
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame read OK: {frame.shape}")
            cv2.imshow("Test", frame)
            print("Window displayed. Press any key...")
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print("✗ Frame read FAILED")
        cap.release()
    else:
        print(f"✗ Camera {cam_idx} FAILED to open")

print("\nDone.")
