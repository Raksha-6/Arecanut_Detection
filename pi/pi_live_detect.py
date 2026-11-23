"""
pi_live_detect.py  -  Live Koleroga / Healthy nut detection on Raspberry Pi
Camera: Raspberry Pi Camera Module 3 (12 MP) via Picamera2 (libcamera).

Setup on the Pi (Raspberry Pi OS Bookworm, 64-bit):
    sudo apt update && sudo apt install -y python3-picamera2
    python3 -m venv --system-site-packages ~/yolo-env
    source ~/yolo-env/bin/activate
    pip install ultralytics ncnn
Copy the exported folder  best_ncnn_model/  next to this script, then:
    python3 pi_live_detect.py
Press q to quit.
"""
import time

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "best_ncnn_model"   # exported with model.export(format="ncnn")
IMGSZ = 640                      # MUST match the imgsz used at export
CONF = 0.35                      # from v2 F1 curve: Koleroga F1 flat 0.3-0.55, lower = fewer missed diseased nuts
FRAME_W, FRAME_H = 1280, 720     # camera preview size (12 MP is far too big for live use)

# BGR colours: green = healthy, red = Koleroga
COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}

model = YOLO(MODEL_PATH, task="detect")
names = model.names

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}))
picam2.start()
# Camera Module 3 has autofocus - keep it in continuous mode
try:
    from libcamera import controls
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
except Exception:
    pass
time.sleep(1)

prev = time.time()
while True:
    # Picamera2 "RGB888" is actually BGR byte order -> already what OpenCV wants
    frame = picam2.capture_array()
    results = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)
    r = results[0]

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = COLORS.get(cls, (255, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)        # red centroid

        if cls == 1:  # Koleroga -> this is where the spray module gets its target
            print(f"KOLEROGA target at X={cx}, Y={cy}  conf={conf:.2f}")

    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Koleroga Detection - Pi", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
