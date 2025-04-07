import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import os

# Setup
os.makedirs("output_frames", exist_ok=True)  # folder to save frames

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # or 'yolov8s.onnx' if you've exported it

# Load COCO class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define parking area polygon
area9 = [(511, 327), (557, 388), (603, 383), (549, 324)]

# Load video
cap = cv2.VideoCapture('parking1.mp4')
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list9 = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])
        class_name = class_list[class_id]

        if class_name == 'car':
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cv2.pointPolygonTest(np.array(area9, np.int32), (cx, cy), False) >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                list9.append(class_name)

    a9 = len(list9)

    if a9 == 1:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, '9', (591, 398), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(area9, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, '9', (591, 398), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Instead of showing frame, save it
    if frame_id % 30 == 0:  # save every 30th frame
        cv2.imwrite(f"output_frames/frame_{frame_id}.jpg", frame)

    frame_id += 1

cap.release()
