"""
prepare_for_roboflow.py
-----------------------
Builds a clean, Roboflow-ready YOLO dataset from:
  - raw images            (new_dataset)
  - auto-generated labels (runs/detect/predict/labels)

What it fixes:
  1. Removes EXACT duplicate images (byte-identical). Roboflow silently drops
     these, which is the main reason your 3026 images shrink on upload.
  2. Strips the 6th "confidence" column that `save_conf=True` added to every
     label line. Proper YOLO labels have exactly 5 numbers per line.
  3. Clamps box coordinates into 0..1 and drops broken lines.
  4. Writes data.yaml so Roboflow maps 0 -> Healthy_nut, 1 -> Koleroga_detector.
  5. Writes review_priority.csv: images sorted so the ones most likely to be
     wrong (no detections / low confidence) come first, so you fix those first.

Only uses the Python standard library. Your original folders are NOT modified.

Usage (Windows CMD, from C:\\Users\\darshan\\Desktop\\Yolo_project):
    python prepare_for_roboflow.py
    python prepare_for_roboflow.py --exclude-nobg     (drop *_nobg images)
"""
import argparse
import csv
import hashlib
import os
import shutil
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "new_dataset")
LBL_DIR = os.path.join(BASE, "runs", "detect", "predict", "labels")
OUT_DIR = os.path.join(BASE, "roboflow_clean")
CLASS_NAMES = ["Healthy_nut", "Koleroga_detector"]  # read from your best.pt
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clamp(v):
    return min(max(v, 0.0), 1.0)


def clean_label(path):
    """Return (clean_lines, confidences, n_bad_lines)."""
    lines, confs, bad = [], [], 0
    if not os.path.exists(path):
        return lines, confs, bad
    with open(path) as f:
        for raw in f:
            p = raw.split()
            if not p:
                continue
            if len(p) not in (5, 6):
                bad += 1
                continue
            try:
                cls = int(float(p[0]))
                x, y, w, h = map(float, p[1:5])
            except ValueError:
                bad += 1
                continue
            if cls not in range(len(CLASS_NAMES)):
                bad += 1
                continue
            # convert to corners, clamp, convert back (keeps boxes inside image)
            x1, y1 = clamp(x - w / 2), clamp(y - h / 2)
            x2, y2 = clamp(x + w / 2), clamp(y + h / 2)
            w, h = x2 - x1, y2 - y1
            if w <= 0.002 or h <= 0.002:
                bad += 1
                continue
            lines.append(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}")
            if len(p) == 6:
                confs.append(float(p[5]))
    return lines, confs, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-nobg", action="store_true",
                    help="skip background-removed *_nobg images")
    args = ap.parse_args()

    if os.path.exists(OUT_DIR):
        raise SystemExit(f"{OUT_DIR} already exists - delete or rename it first.")
    os.makedirs(OUT_DIR)

    images = sorted(f for f in os.listdir(IMG_DIR)
                    if os.path.splitext(f)[1].lower() in IMG_EXT)
    print(f"Images found in new_dataset: {len(images)}")

    # ---- 1. exact duplicates ----
    by_hash = defaultdict(list)
    for i, f in enumerate(images, 1):
        by_hash[md5(os.path.join(IMG_DIR, f))].append(f)
        if i % 500 == 0:
            print(f"  hashed {i}/{len(images)}")
    dup_rows, keep = [], []
    for h, files in by_hash.items():
        keep.append(files[0])
        for d in files[1:]:
            dup_rows.append([d, files[0]])
    print(f"Exact duplicate copies removed: {len(dup_rows)}")

    # ---- 2. copy images + cleaned labels ----
    review, stats = [], defaultdict(int)
    nobg_skipped = 0
    for f in sorted(keep):
        stem = os.path.splitext(f)[0]
        if args.exclude_nobg and stem.lower().endswith("_nobg"):
            nobg_skipped += 1
            continue
        lines, confs, bad = clean_label(os.path.join(LBL_DIR, stem + ".txt"))
        shutil.copy2(os.path.join(IMG_DIR, f), os.path.join(OUT_DIR, f))
        if lines:  # no .txt = Roboflow shows it as "unannotated" -> you label it
            with open(os.path.join(OUT_DIR, stem + ".txt"), "w") as out:
                out.write("\n".join(lines) + "\n")
        for ln in lines:
            stats[CLASS_NAMES[int(ln.split()[0])]] += 1
        stats["images"] += 1
        min_conf = min(confs) if confs else 0.0
        review.append([f, len(lines), round(min_conf, 3),
                       round(sum(confs) / len(confs), 3) if confs else 0.0, bad])

    # ---- 3. data.yaml + reports ----
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as y:
        y.write(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")

    # no detections first, then lowest confidence first
    review.sort(key=lambda r: (r[1] > 0, r[2]))
    with open(os.path.join(BASE, "review_priority.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "num_boxes", "min_conf", "mean_conf", "bad_lines"])
        w.writerows(review)
    with open(os.path.join(BASE, "duplicates_removed.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["duplicate_removed", "identical_to_kept"])
        w.writerows(dup_rows)

    print("\n===== SUMMARY =====")
    print(f"Unique images written to roboflow_clean : {stats['images']}")
    if args.exclude_nobg:
        print(f"_nobg images skipped                     : {nobg_skipped}")
    print(f"Images with NO auto boxes (label by hand): {sum(1 for r in review if r[1] == 0)}")
    print(f"Images with a box below 0.50 confidence  : {sum(1 for r in review if r[1] and r[2] < 0.5)}")
    for c in CLASS_NAMES:
        print(f"Boxes of class {c:<18}: {stats[c]}")
    print("\nNext: upload the whole roboflow_clean folder to Roboflow.")
    print("Reports: duplicates_removed.csv, review_priority.csv")


if __name__ == "__main__":
    main()
