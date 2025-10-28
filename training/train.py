from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

# Train
model.train(
    data="config/cells_aug.yaml",
    epochs=200,
    imgsz=640,
    patience=20,
    batch=8,
    augment=True,
    project="spore_yolov8m_aug",
    name="cells",
)
