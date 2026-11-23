"""
sharp_capture.py  -  Get SHARP 12 MP photos from a vibrating drone (Raspberry Pi 5 + Camera Module 3)

Idea: vibration makes SOME frames blurry, not all. So instead of 1 photo, take a fast
burst of N full-resolution frames and keep only the sharpest one(s).
Sharpness = variance of the Laplacian (more fine edges = higher = sharper).

No detection, no boxes: plain photos.

Run from the Pi DESKTOP terminal (monitor or Pi Connect screen sharing):
  cd ~/koleroga
  ~/yoloenv/bin/python sharp_capture.py
Keys in the live window:
  SPACE = take a burst, keep the sharpest      q = quit

Useful options:
  --burst 8          frames per burst (more = better chance of a sharp one, slower)
  --keep 1           how many of the sharpest frames to save per burst
  --lens 0.7         fixed focus in dioptres = 1 / metres (0.7 ~ 1.5 m, 1.0 = 1 m, 0 = infinity)
  --af               use continuous autofocus instead of fixed focus
  --shutter 1000     fixed shutter in microseconds (1000 = 1/1000 s); 0 = automatic
  --gain 4           brightness boost used with a fixed shutter
  --no-rotate        camera mounted the right way up (default rotates 180 deg)
  --auto-every 3     ALSO take a burst automatically every 3 s (if the Wi-Fi link drops)
  --no-display       no window (use with --auto-every on the drone)
Output: ~/koleroga/field_sharp/<date_time>/  photos + sharpness.csv
"""
import argparse
import csv
import os
import signal
import time

import cv2
from libcamera import Transform, controls
from picamera2 import Picamera2

ap = argparse.ArgumentParser()
ap.add_argument("--burst", type=int, default=8)
ap.add_argument("--keep", type=int, default=1)
ap.add_argument("--lens", type=float, default=0.7)
ap.add_argument("--af", action="store_true")
ap.add_argument("--shutter", type=int, default=0)
ap.add_argument("--gain", type=float, default=4.0)
ap.add_argument("--no-rotate", action="store_true")
ap.add_argument("--auto-every", type=float, default=0)
ap.add_argument("--no-display", action="store_true")
ap.add_argument("--quality", type=int, default=93)
args = ap.parse_args()

RUNNING = True
def _stop(sig, frame):
    global RUNNING
    RUNNING = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

out_dir = os.path.expanduser(f"~/koleroga/field_sharp/{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(out_dir, exist_ok=True)
csv_file = open(os.path.join(out_dir, "sharpness.csv"), "w", newline="")
log = csv.writer(csv_file)
log.writerow(["time", "burst", "file", "sharpness", "rank", "burst_best", "burst_worst",
              "exposure_us", "gain", "lens_position"])

picam2 = Picamera2()
props = picam2.camera_properties
FULL = tuple(props["PixelArraySize"]) if "PixelArraySize" in props else (4608, 2592)
LO = (1280, 720)
tf = Transform() if args.no_rotate else Transform(hflip=1, vflip=1)   # 180 deg rotation
cfg = picam2.create_video_configuration(main={"size": FULL, "format": "RGB888"},
                                        lores={"size": LO, "format": "YUV420"},
                                        transform=tf, buffer_count=4)
picam2.configure(cfg)
picam2.start()

ctrl = {}
if args.af:
    ctrl["AfMode"] = controls.AfModeEnum.Continuous
else:
    ctrl["AfMode"] = controls.AfModeEnum.Manual
    ctrl["LensPosition"] = args.lens
if args.shutter > 0:
    ctrl["AeEnable"] = False
    ctrl["ExposureTime"] = args.shutter
    ctrl["AnalogueGain"] = args.gain
picam2.set_controls(ctrl)
time.sleep(2)
print(f"Saving to {out_dir}   (SPACE = burst, q = quit)")


def sharpness(img):
    small = cv2.resize(img, (1152, 648), interpolation=cv2.INTER_AREA)   # fast + fair comparison
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def do_burst(n_burst):
    frames = []
    for _ in range(args.burst):
        req = picam2.capture_request()
        try:
            img = req.make_array("main")
            md = req.get_metadata()
        finally:
            req.release()
        frames.append((sharpness(img), img, md))
    frames.sort(key=lambda f: f[0], reverse=True)
    best, worst = frames[0][0], frames[-1][0]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for rank, (s, img, md) in enumerate(frames[:args.keep], start=1):
        name = f"{stamp}_b{n_burst:03d}_r{rank}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), n_burst, name, f"{s:.0f}", rank,
                      f"{best:.0f}", f"{worst:.0f}", md.get("ExposureTime"),
                      round(md.get("AnalogueGain", 0), 2), md.get("LensPosition")])
    csv_file.flush()
    print(f"burst {n_burst}: kept sharpness {best:.0f} (worst in burst {worst:.0f}, "
          f"{'big' if best > 1.5 * worst else 'small'} difference)")


n_burst, last_auto = 0, time.time()
while RUNNING:
    take = False
    if not args.no_display:
        req = picam2.capture_request()
        try:
            yuv = req.make_array("lores")
        finally:
            req.release()
        view = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        cv2.putText(view, f"SPACE = sharp photo   q = quit   bursts: {n_burst}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Sharp capture", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        take = key == ord(" ")
    else:
        time.sleep(0.05)
    if args.auto_every > 0 and time.time() - last_auto >= args.auto_every:
        take, last_auto = True, time.time()
    if take:
        n_burst += 1
        do_burst(n_burst)

csv_file.close()
picam2.stop()
if not args.no_display:
    cv2.destroyAllWindows()
print(f"Done: {n_burst} bursts saved in {out_dir}")
