# ===== Google Colab: train YOLOv8n on the cleaned Roboflow dataset =====
# Runtime -> Change runtime type -> T4 GPU. Paste each CELL into its own Colab cell.

# ---- CELL 1: install ----
!pip install -q ultralytics roboflow

# ---- CELL 2: download dataset from Roboflow ----
# In Roboflow: Versions -> your version -> Download Dataset -> format "YOLOv8"
# -> "show download code". Paste YOUR snippet here (it has your api_key):
from roboflow import Roboflow
rf = Roboflow(api_key="PASTE_YOUR_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov8")
print(dataset.location)

# ---- CELL 3: train ----
from ultralytics import YOLO
model = YOLO("yolov8n.pt")            # COCO-pretrained nano = clean start
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    patience=30,                       # early stop if no improvement for 30 epochs
    imgsz=640,                         # nuts in bunch images are small; 640 keeps detail
    batch=16,
    seed=0,
    project="/content/runs", name="koleroga_v2",
)

# ---- CELL 4: honest evaluation on the TEST split (images never used in training) ----
best = YOLO("/content/runs/koleroga_v2/weights/best.pt")
m = best.val(data=f"{dataset.location}/data.yaml", split="test", imgsz=640)
print("mAP50:", m.box.map50, " mAP50-95:", m.box.map)
for i, n in best.names.items():
    print(f"{n:<20} precision={m.box.p[i]:.3f} recall={m.box.r[i]:.3f} mAP50={m.box.ap50[i]:.3f}")
# Also open /content/runs/detect/val*/confusion_matrix.png and PR_curve.png

# ---- CELL 5: export for Raspberry Pi (NCNN is the fastest CPU format on Pi) ----
best.export(format="ncnn", imgsz=640)
!cd /content/runs/koleroga_v2/weights && zip -r best_ncnn_model.zip best_ncnn_model
from google.colab import files
files.download("/content/runs/koleroga_v2/weights/best.pt")
files.download("/content/runs/koleroga_v2/weights/best_ncnn_model.zip")
