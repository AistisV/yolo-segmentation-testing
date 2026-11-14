"""Door / window / fixture detector on 640-px tiles.  python yolo_train.py --data /tmp/yolo/data.yaml --hours 0.45"""
import argparse
from ultralytics import YOLO
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--hours", type=float, default=0.45)
ap.add_argument("--weights", default="yolo11s.pt"); ap.add_argument("--project", default="/vol/ckpt"); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--name", default="yolo")
ap.add_argument("--model", default="")
a = ap.parse_args()
YOLO(a.model or a.weights).train(data=a.data, imgsz=640, epochs=60, time=a.hours, batch=a.batch, workers=8, project=a.project, name=a.name,
                      exist_ok=True, plots=False, degrees=0, fliplr=0.5, flipud=0.5, mosaic=0.5, hsv_h=0, hsv_s=0, hsv_v=0.3)
