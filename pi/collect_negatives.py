"""
collect_negatives.py  -  Hard-negative collector for the Koleroga model (Raspberry Pi 5)

Point the camera at scenes that contain NO arecanuts (faces, empty hands, people,
walls, leaves, trunks, sky, soil, tools...). Every time the model wrongly detects
something, the RAW frame (no boxes drawn) is saved together with an EMPTY .txt
label file. Upload that folder to Roboflow -> they become "null" (background)
images that teach the model: "this is NOT a nut".

Run (from the Pi desktop terminal):
    cd ~/koleroga
    ~/yoloenv/bin/python collect_negatives.py
Press q to quit.  Output: ~/koleroga/negatives/
"""
import os
import time

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "best_ncnn_model"
IMGSZ = 640
CONF = 0.25          # low on purpose: catch even weak false detections
SAVE_EVERY = 1.0     # seconds between saves, avoids 100 near-identical frames
OUT_DIR = os.path.expanduser("~/koleroga/negatives")
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH, task="detect")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}))
picam2.start()
try:
    from libcamera import controls
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
except Exception:
    pass
time.sleep(1)

saved, last_save = 0, 0.0
print("Show the camera things that are NOT nuts. q = quit")
while True:
    frame = picam2.capture_array()
    r = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    view = r.plot()

    now = time.time()
    if len(r.boxes) and now - last_save >= SAVE_EVERY:
        name = time.strftime("neg_%Y%m%d_%H%M%S") + f"_{saved:04d}"
        cv2.imwrite(os.path.join(OUT_DIR, name + ".jpg"), frame)       # raw frame
        open(os.path.join(OUT_DIR, name + ".txt"), "w").close()        # empty label
        saved += 1
        last_save = now

    cv2.putText(view, f"false detections saved: {saved}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Negative collector - NO NUTS in view!", view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
print(f"Saved {saved} hard-negative images to {OUT_DIR}")
