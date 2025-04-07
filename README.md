# Smart_parking_usingYOLOV8

This project uses YOLOv8 object detection to count available parking spaces in a video feed. It defines custom polygonal areas as parking spots and checks if a detected vehicle is inside any of them.

**Files**


testcount.py: Main script that loads a parking video and counts cars in each parking zone using YOLOv8.

coco.txt: Contains the COCO class labels used by YOLOv8 (must include "car").

parking1.mp4: Input parking lot video (you can replace this with your own).

yolov8s.pt: Pre-trained YOLOv8s weights (downloaded from Ultralytics).

**Features**


Detects cars only using YOLOv8.

Calculates the center point of each detection.

Uses 12 custom polygonal parking zones.

Counts how many cars are parked and displays the number of available spaces.

Shows green boundary if space is free, red if occupied.

Real-time visual feedback and coordinates with mouse tracking (for custom zone tuning).

