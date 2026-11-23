"""
pi_detect_record.py  -  Live Koleroga detection on Raspberry Pi 5 WITH recording

Same detection as pi_live_detect.py, plus it saves every session to
~/koleroga/recordings/<date_time>/ :
    raw.mp4          clean camera video (no boxes)  -> reuse for training v4
    annotated.mp4    video with boxes, labels, red centre dots, FPS
    detections.csv   one row per detection: time, frame, class, confidence, box, centre
    snapshots/       photos you save by pressing  s

Run from the Pi DESKTOP terminal (not the remote shell):
    cd ~/koleroga
    ~/yoloenv/bin/python pi_detect_record.py
Keys:  q = stop and save    s = save a snapshot (raw + annotated)
"""
import csv
import os
import time

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "best_ncnn_model"
IMGSZ = 640
CONF = 0.35
FRAME_W, FRAME_H = 1280, 720
COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}   # BGR: green = healthy, red = Koleroga
FPS_WARMUP_FRAMES = 15                      # measure real FPS before starting the video files

# ---------- output folder ----------
session = time.strftime("%Y%m%d_%H%M%S")
out_dir = os.path.expanduser(f"~/koleroga/recordings/{session}")
snap_dir = os.path.join(out_dir, "snapshots")
os.makedirs(snap_dir, exist_ok=True)
csv_file = open(os.path.join(out_dir, "detections.csv"), "w", newline="")
log = csv.writer(csv_file)
log.writerow(["time_s", "frame", "class", "confidence", "x1", "y1", "x2", "y2", "cx", "cy"])

# ---------- model + camera ----------
model = YOLO(MODEL_PATH, task="detect")
names = model.names
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}))
picam2.start()
try:
    from libcamera import controls
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
except Exception:
    pass
time.sleep(1)

raw_writer = ann_writer = None
warmup_times = []
start = prev = time.time()
frame_no = 0
n_kol = n_healthy = 0
print(f"Recording to {out_dir}   (q = stop, s = snapshot)")

while True:
    frame = picam2.capture_array()          # BGR, ready for OpenCV
    raw = frame.copy()                      # clean copy for raw.mp4 / snapshots
    r = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    t = time.time() - start

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
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        log.writerow([f"{t:.2f}", frame_no, names[cls], f"{conf:.3f}", x1, y1, x2, y2, cx, cy])
        if cls == 1:
            n_kol += 1
            print(f"KOLEROGA target at X={cx}, Y={cy}  conf={conf:.2f}")
        else:
            n_healthy += 1

    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}  REC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # open the video files once the real FPS is known, so playback speed is correct
    if raw_writer is None:
        warmup_times.append(now)
        if len(warmup_times) >= FPS_WARMUP_FRAMES:
            rec_fps = max(1.0, (len(warmup_times) - 1) / (warmup_times[-1] - warmup_times[0]))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            raw_writer = cv2.VideoWriter(os.path.join(out_dir, "raw.mp4"), fourcc, rec_fps, (FRAME_W, FRAME_H))
            ann_writer = cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"), fourcc, rec_fps, (FRAME_W, FRAME_H))
            print(f"Video recording started at {rec_fps:.1f} FPS")
    else:
        raw_writer.write(raw)
        ann_writer.write(frame)

    cv2.imshow("Koleroga Detection - Pi (recording)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        name = f"snap_{frame_no:05d}"
        cv2.imwrite(os.path.join(snap_dir, name + "_raw.jpg"), raw)
        cv2.imwrite(os.path.join(snap_dir, name + "_annotated.jpg"), frame)
        print(f"Snapshot saved: {name}")
    frame_no += 1

# ---------- clean shutdown (important: otherwise the .mp4 files are unreadable) ----------
if raw_writer is not None:
    raw_writer.release()
    ann_writer.release()
csv_file.close()
picam2.stop()
cv2.destroyAllWindows()
print(f"\nSaved session to {out_dir}")
print(f"Frames: {frame_no}   Koleroga detections: {n_kol}   Healthy detections: {n_healthy}")
