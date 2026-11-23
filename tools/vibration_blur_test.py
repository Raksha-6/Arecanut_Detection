"""
vibration_blur_test.py  -  Measure how sharp the camera image is (Raspberry Pi 5 + Camera Module 3)

Use it to COMPARE set-ups with numbers instead of guessing:
  motors off vs motors on, old mount vs new damper mount, auto vs fast shutter, etc.

It records N seconds, computes a sharpness score for every frame
(variance of the Laplacian: higher = sharper, blur makes it drop), and prints
average / worst / % of blurry frames. It also saves the sharpest and blurriest frame + a CSV.

Examples (run from the Pi desktop terminal):
  ~/yoloenv/bin/python vibration_blur_test.py --label motors_off
  ~/yoloenv/bin/python vibration_blur_test.py --label motors_on_new_mount
  ~/yoloenv/bin/python vibration_blur_test.py --label fast_shutter --exposure-us 1000 --gain 6
  ~/yoloenv/bin/python vibration_blur_test.py --label fixed_focus_1m --lens 1.0

Options:
  --seconds      how long to record (default 20)
  --exposure-us  fixed shutter time in microseconds (0 = automatic, default)
                 1000 = 1/1000 s, 2000 = 1/500 s. Use outdoors / bright light.
  --gain         analogue gain used with a fixed shutter (default 4.0)
  --lens         fixed focus in dioptres = 1 / distance in metres
                 (1.0 = 1 m, 0.5 = 2 m, 2.0 = 0.5 m). Omit = continuous autofocus.
Keep the SAME scene and lighting for every run you want to compare.
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="test")
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--exposure-us", type=int, default=0)
ap.add_argument("--gain", type=float, default=4.0)
ap.add_argument("--lens", type=float, default=None)
args = ap.parse_args()

out_dir = os.path.expanduser(f"~/koleroga/blur_tests/{time.strftime('%Y%m%d_%H%M%S')}_{args.label}")
os.makedirs(out_dir, exist_ok=True)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)}))
picam2.start()

from libcamera import controls
ctrl = {}
if args.lens is None:
    ctrl["AfMode"] = controls.AfModeEnum.Continuous
else:
    ctrl["AfMode"] = controls.AfModeEnum.Manual
    ctrl["LensPosition"] = args.lens
if args.exposure_us > 0:
    ctrl["AeEnable"] = False
    ctrl["ExposureTime"] = args.exposure_us
    ctrl["AnalogueGain"] = args.gain
picam2.set_controls(ctrl)
time.sleep(2)                                   # let focus / exposure settle

scores, rows = [], []
frames_kept = {}
start = time.time()
while time.time() - start < args.seconds:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s = cv2.Laplacian(gray, cv2.CV_64F).var()
    t = time.time() - start
    scores.append(s)
    rows.append([f"{t:.2f}", f"{s:.1f}"])
    view = frame.copy()
    cv2.putText(view, f"sharpness {s:.0f}  {args.label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Blur test (q = stop early)", view)
    if "best" not in frames_kept or s > frames_kept["best"][0]:    # remember sharpest frame
        frames_kept["best"] = (s, frame)
    if "worst" not in frames_kept or s < frames_kept["worst"][0]:  # remember blurriest frame
        frames_kept["worst"] = (s, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

meta = picam2.capture_metadata()
picam2.stop()
cv2.destroyAllWindows()

scores = np.array(scores)
median = float(np.median(scores))
blurry = float(np.mean(scores < 0.5 * median) * 100)   # frames under half the median = clearly blurred
with open(os.path.join(out_dir, "sharpness.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["time_s", "sharpness"]); w.writerows(rows)
cv2.imwrite(os.path.join(out_dir, "blurriest.jpg"), frames_kept["worst"][1])
cv2.imwrite(os.path.join(out_dir, "sharpest.jpg"), frames_kept["best"][1])

print(f"\n===== {args.label} =====")
print(f"frames: {len(scores)}   fps: {len(scores) / max(1e-6, (time.time() - start)):.1f}")
print(f"sharpness  mean {scores.mean():.0f}   median {median:.0f}   worst {scores.min():.0f}   best {scores.max():.0f}")
print(f"spread (std / mean): {scores.std() / scores.mean():.2f}   (lower = steadier image)")
print(f"clearly blurred frames (< half of median): {blurry:.1f} %")
print(f"camera reported: ExposureTime {meta.get('ExposureTime')} us, AnalogueGain {meta.get('AnalogueGain', 0):.1f}, "
      f"LensPosition {meta.get('LensPosition')}")
print(f"saved to {out_dir}")
