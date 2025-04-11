# Smart_parking_usingYOLOV8

This project uses YOLOv8 object detection to count available parking spaces in a video feed. It defines custom polygonal areas as parking spots and checks if a detected vehicle is inside any of them.

## Files


testcount.py: Main script that loads a parking video and counts cars in each parking zone using YOLOv8.

coco.txt: Contains the COCO class labels used by YOLOv8 (must include "car").

parking1.mp4: Input parking lot video (you can replace this with your own).

yolov8s.pt: Pre-trained YOLOv8s weights (downloaded from Ultralytics).


## Results

The system shows the following in real-time:

Parking slots with red outlines = Occupied

Parking slots with green outlines = Available

Display of total Available Slots at the top-left corner



![frame_120](https://github.com/user-attachments/assets/54e50e95-acde-430e-a111-d9ef83fe7b3e)


![frame_60](https://github.com/user-attachments/assets/833cd61a-e50e-4517-901d-ccd41bc616a8)


✅ Requirements
Python 3.x

YOLOv8 by Ultralytics

OpenCV – For image processing and drawing.

NumPy & Pandas – For array and tabular data handling.



## Features


Detects cars only using YOLOv8.

Calculates the center point of each detection.

Uses 12 custom polygonal parking zones.

Counts how many cars are parked and displays the number of available spaces.

Shows green boundary if space is free, red if occupied.

Real-time visual feedback and coordinates with mouse tracking (for custom zone tuning).

