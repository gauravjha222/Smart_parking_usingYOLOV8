
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os

# Create folder to save frames
os.makedirs("output_testcount", exist_ok=True)

# Load YOLO model
model = YOLO('yolov8s.pt')  # You can change to 'yolov8s.onnx' if using ONNX

# Load class names
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Video input
cap = cv2.VideoCapture('parking1.mp4')
frame_id = 0

# Define parking polygons
areas = [
    [(52,364),(30,417),(73,412),(88,369)],
    [(105,353),(86,428),(137,427),(146,358)],
    [(159,354),(150,427),(204,425),(203,353)],
    [(217,352),(219,422),(273,418),(261,347)],
    [(274,345),(286,417),(338,415),(321,345)],
    [(336,343),(357,410),(409,408),(382,340)],
    [(396,338),(426,404),(479,399),(439,334)],
    [(458,333),(494,397),(543,390),(495,330)],
    [(511,327),(557,388),(603,383),(549,324)],
    [(564,323),(615,381),(654,372),(596,315)],
    [(616,316),(666,369),(703,363),(642,312)],
    [(674,311),(730,360),(764,355),(707,308)]
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    occupied = [0] * len(areas)

    for _, row in px.iterrows():
        x1, y1, x2, y2, _, cls_id = map(int, row[:6])
        cls = class_list[cls_id]
        if cls == 'car':
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            for idx, area in enumerate(areas):
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    occupied[idx] = 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # Count and display
    total = len(areas)
    count_occupied = sum(occupied)
    count_empty = total - count_occupied

    for i, area in enumerate(areas):
        color = (0, 0, 255) if occupied[i] else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(frame, str(i+1), (area[0][0], area[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Add total available display
    cv2.putText(frame, f"Available: {count_empty}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Save every 30th frame
    if frame_id % 30 == 0:
        cv2.imwrite(f"output_testcount/frame_{frame_id}.jpg", frame)

    frame_id += 1

cap.release()


