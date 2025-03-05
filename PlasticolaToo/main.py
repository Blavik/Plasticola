# Image processor + YOLOv8
import cv2
import numpy as np
import json
import time
from ultralytics import YOLO

# Set up the YOLO model
model = YOLO("yolov8n.pt")  # YOLOv8 Nano - Optimized for Raspberry Pi

# Manually set camera device paths (confirmed working, changing cameras could break code)
camera_paths = ["/dev/video0", "/dev/video2", "/dev/video4", "/dev/video6"]
camera_labels = ["left", "front", "right", "back"]  # Assign readable names
camera_width, camera_height = 320, 240  # native camera resolution (320*240)
full_width, full_height = camera_width * 4, camera_height  # Full stitched resolution (1280*240)

# Initialize cameras
cams = [cv2.VideoCapture(path) for path in camera_paths]

# Check if cameras are connected
for path, cam in zip(camera_paths, cams):
    if not cam.isOpened():
        print(f"Camera {path} is not connected! Exiting.")
        exit()

# Capture a shot on all 4 cameras
frames = []
for cam in cams:
    ret, frame = cam.read()
    if ret:
        frames.append(frame)

# Release cameras
for cam in cams:
    cam.release()

# Ensure all frames are captured before proceeding
if len(frames) != 4:
    print("Error: Not all cameras captured correctly.")
    exit()

# Resize all frames to 320x240
frames_resized = [cv2.resize(frame, (camera_width, camera_height)) for frame in frames]

# Combine all frames into one horizontal image
combined_image = np.hstack(frames_resized)

# Save the combined image
cv2.imwrite('combined_image.jpg', combined_image)

# Perform YOLOv8 inference on the combined image
start_time = time.time()
results = model.predict(combined_image, verbose=False)
inference_time = time.time() - start_time

# Extract bounding boxes and class IDs
bounding_boxes = results[0].boxes.xywh.tolist()  # Get (x_center, y_center, width, height)
class_ids = results[0].boxes.cls.tolist()  # Get class IDs (only count "person" detections)

# Calculate total image area (1280x240 = 307200 pixels)
total_img_area = full_width * full_height

# Initialize per-camera coverage
camera_coverage = {side: 0.0 for side in camera_labels}
total_human_coverage = 0.0
total_humans_detected = 0

# Process bounding boxes
for (x, y, w, h), class_id in zip(bounding_boxes, class_ids):
    if int(class_id) == 0:  # Only consider humans (YOLO class 0)
        total_humans_detected += 1  # Count humans

        # Determine which camera the human is in
        if 0 <= x < 320:
            camera = "left"
        elif 320 <= x < 640:
            camera = "front"
        elif 640 <= x < 960:
            camera = "right"
        else:
            camera = "back"

        # Normalize bounding box size relative to **its own camera's 320x240 area**
        side_img_area = camera_width * camera_height
        coverage = (w * h) / side_img_area

        # Accumulate per-camera coverage
        camera_coverage[camera] += coverage

        # Normalize total coverage relative to **the full 1280x240 area**
        total_human_coverage += (w * h) / total_img_area

# Save results to JSON file
data = {
    "camera_coverage": camera_coverage,  # Per-side coverage (0.0 - 1.0)
    "total_human_coverage": total_human_coverage,  # Full 1280x240 coverage (0.0 - 1.0)
    "total_humans_detected": total_humans_detected,  # Number of detected humans
    "inference_time": inference_time  # Track inference delay
}

with open("output.json", "w") as f:
    json.dump(data, f, indent=4)

print(f"Detection complete! Coverage & human count saved to output.json (Inference Time: {inference_time:.4f} sec)")
