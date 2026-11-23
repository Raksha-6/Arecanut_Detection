"""
pi_detect_v2.py  -  Improved live Koleroga detection for Raspberry Pi 5 + Camera Module 3

New compared to pi_live_detect.py
  1. ONE box per nut      : agnostic NMS, no more Healthy + Koleroga boxes on the same nut
  2. TRACKING + VOTING    : every nut gets an ID; its label is decided by the last N frames,
                            so it stops flickering. "KOLEROGA CONFIRMED" only after K of N votes.
  3. DIGITAL ZOOM         : uses the full 12 MP sensor and crops the centre -> more pixels per nut
                            (better range). --zoom 1 = full view, 2 = 2x (half the view width)
  4. FIXED FOCUS          : --focus-m 1.0 locks focus at 1 m (no autofocus hunting)
  5. FAST SHUTTER         : --exposure-us 1000 = 1/1000 s (less motion blur, needs daylight)
  6. BLUR FILTER          : very blurred frames are shown but NOT used for votes
  7. OPTIONAL RECORDING   : --record saves raw.mp4, annotated.mp4, detections.csv

Run from the Pi DESKTOP terminal:
  cd ~/koleroga
  ~/yoloenv/bin/python pi_detect_v2.py                              # defaults
  ~/yoloenv/bin/python pi_detect_v2.py --zoom 2 --focus-m 1.0       # range test
  ~/yoloenv/bin/python pi_detect_v2.py --zoom 2 --focus-m 1.5 --exposure-us 1000 --record
On the drone (no monitor), keeps running if the remote shell disconnects:
  nohup ~/yoloenv/bin/python pi_detect_v2.py --record --no-display > ~/koleroga/detect.log 2>&1 &
  stop:  pkill -INT -f pi_detect_v2.py
Keys: q = quit (window mode)
"""
import argparse
import csv
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="best_ncnn_model")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--zoom", type=float, default=1.0, help="1 = full view, 2 = 2x digital zoom")
ap.add_argument("--focus-m", type=float, default=None, help="fixed focus distance in metres (default: autofocus)")
ap.add_argument("--exposure-us", type=int, default=0, help="fixed shutter in microseconds (0 = auto)")
ap.add_argument("--gain", type=float, default=4.0, help="analogue gain when shutter is fixed")
ap.add_argument("--window", type=int, default=10, help="frames used for voting per nut")
ap.add_argument("--confirm", type=int, default=7, help="Koleroga votes needed (out of --window) to confirm")
ap.add_argument("--blur-ratio", type=float, default=0.4, help="frame ignored if sharpness < ratio x recent median")
ap.add_argument("--record", action="store_true")
ap.add_argument("--no-display", action="store_true",
                help="no video window (on the drone / remote shell). Stop with: pkill -INT -f pi_detect_v2.py")
args = ap.parse_args()

# stop cleanly on Ctrl+C or pkill (so the .mp4 files are closed properly)
import signal
RUNNING = True
def _stop(sig, frame):
    global RUNNING
    RUNNING = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

FRAME_W, FRAME_H = 1280, 720
KOL = 1                                          # class id of Koleroga_detector
GREEN, RED, GREY, YELLOW = (0, 200, 0), (0, 0, 255), (160, 160, 160), (0, 255, 255)

# ---------------- camera ----------------
picam2 = Picamera2()
def sensor_size(cam):
    """Full sensor resolution (4608 x 2592 on Camera Module 3), works on old and new Picamera2/libcamera."""
    props = cam.camera_properties
    if "PixelArraySize" in props:
        return tuple(props["PixelArraySize"])
    try:
        return tuple(cam.sensor_resolution)
    except Exception:
        return (4608, 2592)


full_w, full_h = sensor_size(picam2)
print(f"Sensor size: {full_w} x {full_h}")
if args.zoom > 1.0:
    # full-resolution sensor mode, so zooming adds REAL detail (not just upscaling); ~14 fps max
    try:
        cfg = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)},
                                                  sensor={"output_size": (full_w, full_h)})
    except TypeError:                                                  # older Picamera2 versions
        cfg = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)},
                                                  raw={"size": (full_w, full_h)})
else:
    cfg = picam2.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)})
picam2.configure(cfg)
picam2.start()

from libcamera import controls
ctrl = {}
if args.zoom > 1.0:
    # crop the centre of the largest area the camera allows (its own coordinates, version-safe)
    try:
        x0, y0, mw, mh = picam2.camera_controls["ScalerCrop"][1]
    except Exception:
        x0, y0, mw, mh = 0, 0, full_w, full_h
    cw, ch = int(mw / args.zoom), int(mh / args.zoom)
    ctrl["ScalerCrop"] = (x0 + (mw - cw) // 2, y0 + (mh - ch) // 2, cw, ch)
    print(f"Digital zoom {args.zoom:g}x: crop {cw} x {ch} of {mw} x {mh}")
if args.focus_m:
    ctrl["AfMode"] = controls.AfModeEnum.Manual
    ctrl["LensPosition"] = 1.0 / args.focus_m                        # dioptres
else:
    ctrl["AfMode"] = controls.AfModeEnum.Continuous
if args.exposure_us > 0:
    ctrl["AeEnable"] = False
    ctrl["ExposureTime"] = args.exposure_us
    ctrl["AnalogueGain"] = args.gain
picam2.set_controls(ctrl)
time.sleep(2)

# ---------------- model ----------------
model = YOLO(args.model, task="detect")
names = model.names

# ---------------- recording (optional) ----------------
writers, log, csv_file, out_dir = None, None, None, None
if args.record:
    out_dir = os.path.expanduser(f"~/koleroga/recordings/{time.strftime('%Y%m%d_%H%M%S')}_v2_zoom{args.zoom:g}")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = open(os.path.join(out_dir, "detections.csv"), "w", newline="")
    log = csv.writer(csv_file)
    log.writerow(["time_s", "frame", "track_id", "frame_class", "conf", "voted_class",
                  "kol_votes", "votes", "cx", "cy", "x1", "y1", "x2", "y2"])
    print(f"Recording to {out_dir}")

votes = defaultdict(lambda: deque(maxlen=args.window))   # track_id -> recent class ids
last_seen = {}
confirmed = set()
sharp_hist = deque(maxlen=30)
warm_t, start, prev = [], time.time(), time.time()
frame_no = 0

while True:
    frame = picam2.capture_array()
    raw = frame.copy()
    now = time.time()

    # ---- blur check ----
    sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    med = float(np.median(sharp_hist)) if len(sharp_hist) >= 5 else sharp
    sharp_hist.append(sharp)
    blurry = sharp < args.blur_ratio * med

    # ---- detect + track ----
    r = model.track(frame, imgsz=args.imgsz, conf=args.conf, agnostic_nms=True,
                    persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    if r.boxes is not None and len(r.boxes):
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for box, tid in zip(r.boxes, ids):
            cls = int(box.cls[0]); conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if tid is not None and not blurry:
                votes[tid].append(cls)
                last_seen[tid] = frame_no
            v = votes.get(tid, [])
            n_kol = sum(1 for c in v if c == KOL)
            n = len(v)
            if tid is None:                             # tracker has no ID yet -> plain detection colour
                voted = cls
                color = RED if cls == KOL else GREEN
            elif n >= 3:                                # enough history -> majority vote
                voted = KOL if n_kol > n / 2 else 0
                color = RED if voted == KOL else GREEN
            else:                                       # too new to decide -> grey "checking"
                voted = None
                color = GREY

            is_conf = tid is not None and n_kol >= args.confirm
            if is_conf and tid not in confirmed:
                confirmed.add(tid)
                print(f"KOLEROGA CONFIRMED  id={tid}  X={cx} Y={cy}  ({n_kol}/{n} votes)")

            tag = names[voted] if voted is not None else "checking"
            label = f"#{tid} {tag} {n_kol}/{n}" if tid is not None else f"{names[cls]} {conf:.2f}"
            if is_conf:
                label += " CONFIRMED"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_conf else 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            if log:
                log.writerow([f"{now - start:.2f}", frame_no, tid, names[cls], f"{conf:.3f}",
                              names[voted] if voted is not None else "", n_kol, n, cx, cy, x1, y1, x2, y2])

    # forget nuts not seen for a while
    for tid in [t for t, f in last_seen.items() if frame_no - f > 60]:
        votes.pop(tid, None); last_seen.pop(tid, None)

    fps = 1.0 / max(now - prev, 1e-6); prev = now
    status = f"FPS {fps:.1f}  zoom {args.zoom:g}x  sharp {sharp:.0f}" + ("  BLURRY-skipped" if blurry else "")
    if args.record:
        status += "  REC"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED if blurry else YELLOW, 2)

    # ---- recording ----
    if args.record:
        if writers is None:
            warm_t.append(now)
            if len(warm_t) >= 15:
                rec_fps = max(1.0, (len(warm_t) - 1) / (warm_t[-1] - warm_t[0]))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writers = (cv2.VideoWriter(os.path.join(out_dir, "raw.mp4"), fourcc, rec_fps, (FRAME_W, FRAME_H)),
                           cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"), fourcc, rec_fps, (FRAME_W, FRAME_H)))
                print(f"Video recording started at {rec_fps:.1f} FPS")
        else:
            writers[0].write(raw); writers[1].write(frame)

    if not args.no_display:
        cv2.imshow("Koleroga Detection v2", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if not RUNNING:
        break
    frame_no += 1

if writers:
    writers[0].release(); writers[1].release()
if csv_file:
    csv_file.close()
picam2.stop()
if not args.no_display:
    cv2.destroyAllWindows()
print(f"\nFrames: {frame_no}   Koleroga confirmed (unique nuts): {len(confirmed)}")
if out_dir:
    print(f"Saved to {out_dir}")
