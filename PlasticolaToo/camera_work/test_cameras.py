import cv2 # Image processor
import numpy as np # Python library

# Manually set camera device paths
# (Paths might change if cameras are swapped out!)
camera_paths = ["/dev/video0", "/dev/video2", "/dev/video4", "/dev/video6"]

# Initialize cameras
cams = [cv2.VideoCapture(path) for path in camera_paths]

# Check if cameras are connected
for path, cam in zip(camera_paths, cams):
    if not cam.isOpened():
        print(f"Camera {path} is not connected! :/ Exiting.")
        exit()

# Capture a shot on all 4 cameras
frames = [cam.read()[1] for cam in cams if cam.isOpened()]

# Release cameras
for cam in cams:
    cam.release()

# Resize all frames to 320x240
frames_resized = [cv2.resize(frame, (320, 240)) for frame in frames]

# Combine and save image
cv2.imwrite('combined_image_test.jpg', np.hstack(frames_resized))

print("Images successfully captured and saved! Check 'combined_image_test.jpg'")
