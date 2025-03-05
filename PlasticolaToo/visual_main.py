import cv2
import numpy as np
import json
import time
import threading
from ultralytics import YOLO

# Set up YOLO model
model = YOLO("yolov8n.pt")  # YOLOv8 Nano - Optimized for Raspberry Pi

# Manually set camera device paths (confirmed working)
camera_paths = ["/dev/video0", "/dev/video2", "/dev/video4", "/dev/video6"]
camera_labels = ["left", "back", "right", "front"]  # Swapped front & back labels
camera_width, camera_height = 320, 240  # Each camera resolution
full_width, full_height = camera_width * 4, camera_height  # Full stitched resolution

# Scale factor for display size (keeps aspect ratio)
scale_factor = 2  # Adjust this to increase/decrease the displayed size
new_width = full_width * scale_factor
new_height = full_height * scale_factor

# Initialize cameras
cams = [cv2.VideoCapture(path) for path in camera_paths]

# Set camera properties for better balance between speed and quality
for cam in cams:
    cam.set(cv2.CAP_PROP_FPS, 30)  # Reduce FPS target for stability
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering lag

# Check if cameras are connected
for path, cam in zip(camera_paths, cams):
    if not cam.isOpened():
        print(f"Camera {path} is not connected! Exiting.")
        exit()

# Create a resizable window
cv2.namedWindow("360-Degree Human Detection", cv2.WINDOW_NORMAL)

last_json_save = time.time()  # Track time for JSON saving
frame_lock = threading.Lock()
latest_frame = None

def capture_frames():
    global latest_frame
    while True:
        frames = []
        for cam in cams:
            ret, frame = cam.read()
            if ret:
                frames.append(frame)
        
        if len(frames) == 4:
            frames_resized = [cv2.resize(frame, (camera_width, camera_height), interpolation=cv2.INTER_LINEAR) for frame in frames]
            with frame_lock:
                latest_frame = np.hstack(frames_resized)

# Start capturing frames in a separate thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

while True:
    start_time = time.time()  # Measure FPS
    
    with frame_lock:
        if latest_frame is None:
            continue
        combined_image = latest_frame.copy()
    
    # Perform YOLOv8 inference
    inference_start = time.time()
    results = model.predict(combined_image, verbose=False)  # Removed FP16 optimization for stability
    inference_time = time.time() - inference_start

    # Extract bounding boxes and class IDs
    bounding_boxes = results[0].boxes.xywh.tolist()  # (x_center, y_center, width, height)
    class_ids = results[0].boxes.cls.tolist()  # Get class IDs (only count "person")

    # Initialize per-camera coverage
    camera_coverage = {side: 0.0 for side in camera_labels}
    total_human_coverage = 0.0
    total_humans_detected = 0

    # Draw bounding boxes
    for (x, y, w, h), class_id in zip(bounding_boxes, class_ids):
        if int(class_id) == 0:  # Only process humans (YOLO class 0)
            total_humans_detected += 1

            # Determine which camera (side) detected the human
            if 0 <= x < 320:
                camera = "left"
            elif 320 <= x < 640:
                camera = "back"  # Swapped front & back
            elif 640 <= x < 960:
                camera = "right"
            else:
                camera = "front"  # Swapped front & back

            # Normalize bounding box size relative to **its own camera's 320x240 area**
            side_img_area = camera_width * camera_height
            coverage = (w * h) / side_img_area
            camera_coverage[camera] += coverage

            # Normalize total coverage relative to **the full 1280x240 area**
            total_human_coverage += (w * h) / (full_width * full_height)

            # Draw bounding box on the image
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with camera side
            cv2.putText(combined_image, camera, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display JSON data on the image
    overlay_text = f"Humans: {total_humans_detected} | Total Coverage: {total_human_coverage:.2f} | Inference: {inference_time:.3f}s"
    cv2.putText(combined_image, overlay_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y_offset = 40
    for side, coverage in camera_coverage.items():
        text = f"{side.capitalize()} Coverage: {coverage:.2f}"
        cv2.putText(combined_image, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 20

    # **Resize Image for Larger Display Without Distortion**
    display_image = cv2.resize(combined_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Show the resized image
    cv2.imshow("360-Degree Human Detection", display_image)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print FPS
    fps = 1 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

# Release cameras and close OpenCV window
for cam in cams:
    cam.release()
cv2.destroyAllWindows()
