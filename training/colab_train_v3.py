# ===== Google Colab: train the v3 model (Roboflow dataset + background images) =====
# Runtime -> Change runtime type -> T4 GPU. Paste each CELL into its own Colab cell.
# v3 = v2 data + 105 new Koleroga images (Roboflow) + 544 background images (no nuts)
# that are added here, outside Roboflow, with empty label files.

# ---- CELL 1: install + Google Drive ----
!pip install -q ultralytics roboflow
from google.colab import drive
drive.mount('/content/drive')

# ---- CELL 2: download the Roboflow dataset (YOLOv8 format) ----
# Get YOUR snippet from Roboflow: Versions -> Download Dataset -> YOLOv8 -> show download code.
# Never commit a real API key to GitHub.
from roboflow import Roboflow
import os
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(3).download("yolov8")
total = sum(len(os.listdir(f"{dataset.location}/{s}/images")) for s in ["train", "valid", "test"])
print("Dataset at:", dataset.location, "| images:", total)

# ---- CELL 3: add background images (zip in My Drive, images + empty .txt files) ----
import zipfile, glob, random, shutil
zip_path = "/content/drive/MyDrive/background_v3.zip"
bg_dir = "/content/bg"
with zipfile.ZipFile(zip_path) as z:
    z.extractall(bg_dir)
imgs = sorted(p for p in glob.glob(f"{bg_dir}/**/*", recursive=True)
              if p.lower().endswith((".jpg", ".jpeg", ".png")))
random.seed(0); random.shuffle(imgs)
n = len(imgs); n_tr = int(0.7 * n); n_va = int(0.2 * n)
splits = {"train": imgs[:n_tr], "valid": imgs[n_tr:n_tr + n_va], "test": imgs[n_tr + n_va:]}
for split, files in splits.items():
    for p in files:
        folder = os.path.basename(os.path.dirname(p))
        stem, ext = os.path.splitext(os.path.basename(p))
        new = f"bg_{folder}_{stem}"
        shutil.copy(p, f"{dataset.location}/{split}/images/{new}{ext.lower()}")
        open(f"{dataset.location}/{split}/labels/{new}.txt", "w").close()   # empty = no nut
    print(f"{split}: added {len(files)} background images")

# ---- CELL 4: train ----
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data=f"{dataset.location}/data.yaml", epochs=100, patience=30, imgsz=640,
            batch=16, seed=0, project="/content/drive/MyDrive/koleroga_runs", name="koleroga_v3")
# NOTE: if the folder already exists Ultralytics saves to koleroga_v3-2 etc. Check the log line
# "Results saved to ..." and use that path below.
RUN = "/content/drive/MyDrive/koleroga_runs/koleroga_v3"

# ---- CELL 5: evaluate (per class) + false alarms on non-nut images ----
w = f"{RUN}/weights/best.pt"
m = YOLO(w)
r = m.val(data=f"{dataset.location}/data.yaml", split="val", imgsz=640, plots=True)
for i, name in m.names.items():
    print(f"{name:<20} precision={r.box.p[i]:.3f} recall={r.box.r[i]:.3f} mAP50={r.box.ap50[i]:.3f}")
bg_val = glob.glob(f"{dataset.location}/valid/images/bg_*")
hits = sum(1 for p in m.predict(bg_val, conf=0.35, imgsz=640, stream=True, verbose=False) if len(p.boxes))
print(f"False alarms on non-nut images: {hits}/{len(bg_val)}")

# ---- CELL 6: export for Raspberry Pi (NCNN) + download ----
m.export(format="ncnn", imgsz=640)
!cd {RUN}/weights && zip -qr /content/best_ncnn_model.zip best_ncnn_model
from google.colab import files
files.download(w)
files.download("/content/best_ncnn_model.zip")
