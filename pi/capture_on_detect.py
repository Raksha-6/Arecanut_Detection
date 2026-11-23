"""
capture_on_detect.py  -  Save FULL 12 MP photos only when the model sees arecanut
Raspberry Pi 5 + Camera Module 3, for field data collection (manual Loiter flights).

How it works
  - The camera runs two streams at once from the full sensor:
      lores 1280x720  -> the model looks at this (fast)
      main 4608x2592  -> saved as a JPEG only when something is detected
  - A trigger = at least one box (Healthy_nut or Koleroga_detector) with confidence >= --conf
  - At most one triggered photo every --min-gap seconds (avoids 100 identical photos)
  - --interval N also saves one photo every N seconds even with no detection
    (backup, so you still get data if the model misses real bunches)
  - Saving happens in a background thread, so detection keeps running

Output: ~/koleroga/field_captures/<date_time>/
    full/      12 MP photos  (training data for v4 and ground-station analysis)
    preview/   small 1280x720 copies with boxes drawn (quick check what triggered)
    captures.csv  time, file, reason, boxes, classes, confidences, exposure, focus

Run:
  With a screen:   ~/yoloenv/bin/python capture_on_detect.py
  On the drone:    nohup ~/yoloenv/bin/python capture_on_detect.py --no-display --interval 5 > ~/koleroga/capture.log 2>&1 &
  Stop (drone):    pkill -INT -f capture_on_detect.py
Options: --conf 0.3  --min-gap 1.0  --interval 0  --focus-m 1.5  --exposure-us 1000 --gain 4  --no-display
"""
import argparse
import csv
import os
import queue
import signal
import threading
import time

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="best_ncnn_model")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--conf", type=float, default=0.3, help="min confidence that triggers a photo")
ap.add_argument("--min-gap", type=float, default=1.0, help="seconds between triggered photos")
ap.add_argument("--interval", type=float, default=0, help="also save a photo every N s (0 = off)")
ap.add_argument("--focus-m", type=float, default=None, help="fixed focus in metres (default: autofocus)")
ap.add_argument("--exposure-us", type=int, default=0, help="fixed shutter in microseconds (0 = auto)")
ap.add_argument("--gain", type=float, default=4.0)
ap.add_argument("--quality", type=int, default=92, help="JPEG quality")
ap.add_argument("--no-display", action="store_true")
args = ap.parse_args()

RUNNING = True
def _stop(sig, frame):
    global RUNNING
    RUNNING = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ---------------- output ----------------
out_dir = os.path.expanduser(f"~/koleroga/field_captures/{time.strftime('%Y%m%d_%H%M%S')}")
full_dir, prev_dir = os.path.join(out_dir, "full"), os.path.join(out_dir, "preview")
os.makedirs(full_dir, exist_ok=True)
os.makedirs(prev_dir, exist_ok=True)
csv_file = open(os.path.join(out_dir, "captures.csv"), "w", newline="")
log = csv.writer(csv_file)
log.writerow(["time", "file", "reason", "n_boxes", "classes", "confidences",
              "boxes_full_res_xyxy", "exposure_us", "gain", "lens_position"])

# ---------------- camera: full-res main + small lores ----------------
picam2 = Picamera2()
props = picam2.camera_properties
if "PixelArraySize" in props:
    FULL = tuple(props["PixelArraySize"])
else:
    try:
        FULL = tuple(picam2.sensor_resolution)
    except Exception:
        FULL = (4608, 2592)
LO = (1280, 720)
cfg = picam2.create_video_configuration(main={"size": FULL, "format": "RGB888"},
                                        lores={"size": LO, "format": "YUV420"},
                                        buffer_count=3)
picam2.configure(cfg)
picam2.start()

from libcamera import controls
ctrl = {}
if args.focus_m:
    ctrl["AfMode"] = controls.AfModeEnum.Manual
    ctrl["LensPosition"] = 1.0 / args.focus_m
else:
    ctrl["AfMode"] = controls.AfModeEnum.Continuous
if args.exposure_us > 0:
    ctrl["AeEnable"] = False
    ctrl["ExposureTime"] = args.exposure_us
    ctrl["AnalogueGain"] = args.gain
picam2.set_controls(ctrl)
time.sleep(2)
print(f"Camera: full {FULL[0]}x{FULL[1]}, detection on {LO[0]}x{LO[1]}. Saving to {out_dir}")

model = YOLO(args.model, task="detect")
names = model.names
scale = FULL[0] / LO[0]

# ---------------- background saver ----------------
jobs = queue.Queue(maxsize=4)
def saver():
    while True:
        job = jobs.get()
        if job is None:
            break
        path, img = job
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        jobs.task_done()
t_saver = threading.Thread(target=saver, daemon=True)
t_saver.start()

n_saved = n_trig = n_int = n_dropped = 0
last_trig = last_int = 0.0
prev = time.time()

while RUNNING:
    req = picam2.capture_request()
    try:
        yuv = req.make_array("lores")
        md = req.get_metadata()
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        r = model.predict(bgr, imgsz=args.imgsz, conf=args.conf, agnostic_nms=True, verbose=False)[0]
        now = time.time()

        reason = None
        if len(r.boxes) and now - last_trig >= args.min_gap:
            reason = "detected"
        elif args.interval > 0 and now - last_int >= args.interval:
            reason = "interval"

        view = r.plot() if len(r.boxes) else bgr
        if reason:
            stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((now % 1) * 1000):03d}"
            name = f"{stamp}_{reason}.jpg"
            if jobs.full():
                n_dropped += 1                      # SD card too slow: skip instead of running out of memory
            else:
                full = req.make_array("main")       # copy of the 12 MP frame
                jobs.put((os.path.join(full_dir, name), full))
                cv2.imwrite(os.path.join(prev_dir, name), view, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cls = [names[int(c)] for c in r.boxes.cls.tolist()]
                confs = [f"{c:.2f}" for c in r.boxes.conf.tolist()]
                boxes = [[int(v * scale) for v in b] for b in r.boxes.xyxy.tolist()]
                log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, reason, len(cls),
                              ";".join(cls), ";".join(confs), ";".join(",".join(map(str, b)) for b in boxes),
                              md.get("ExposureTime"), round(md.get("AnalogueGain", 0), 2), md.get("LensPosition")])
                csv_file.flush()
                n_saved += 1
                if reason == "detected":
                    n_trig += 1; last_trig = now
                else:
                    n_int += 1; last_int = now
                print(f"saved {name}  ({len(cls)} boxes)")
    finally:
        req.release()

    if not args.no_display:
        fps = 1.0 / max(now - prev, 1e-6)
        cv2.putText(view, f"FPS {fps:.1f}  saved {n_saved} (detected {n_trig}, interval {n_int})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Capture on detect (q = stop)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    prev = now

# ---------------- clean stop ----------------
print("Stopping, finishing photo saves...")
jobs.join()
jobs.put(None)
t_saver.join(timeout=10)
csv_file.close()
picam2.stop()
if not args.no_display:
    cv2.destroyAllWindows()
print(f"Done. Saved {n_saved} photos ({n_trig} detected, {n_int} interval), dropped {n_dropped}. Folder: {out_dir}")
