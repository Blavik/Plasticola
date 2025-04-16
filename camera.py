import cv2
import numpy as np
import json
import time
import os
import threading
import shutil
from collections import deque
from ultralytics import YOLO

# ───── Config ───── #
camera_paths = ["/dev/video0", "/dev/video2", "/dev/video4", "/dev/video6"]
camera_labels = ["left", "back", "right", "front"]
camera_width, camera_height = 480, 272
frame_width, frame_height = 1280, 720
grid_cols, grid_rows = 2, 2
full_width, full_height = camera_width * grid_cols, camera_height * grid_rows
model_path = "yolov8n.pt"
conf_threshold = 0.25
json_output_path = "coverage.json"
show_preview = True  # Toggle preview window

# ───── Global Shared Variables ───── #
stitched_frame = None
inference_boxes = []
coverage_data = {}
inference_times = deque(maxlen=300)

frame_lock = threading.Lock()
inference_lock = threading.Lock()
stop_event = threading.Event()

# ───── Setup Cameras ───── #
cams = []
for path in camera_paths:
    cam = cv2.VideoCapture(path)
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cam.set(cv2.CAP_PROP_FPS, 10)
        cams.append(cam)
    else:
        cams.append(None)

# ───── Load & Fuse the Model ───── #
model = YOLO(model_path)
model.fuse()

# ───── Console Output Utilities ───── #
def clear_console_lines(n=20):
    print("\033[F" * n, end="")  # Move cursor up n lines

def print_static_output(inf_time, post_time, num_boxes, avg_fps, coverage_dict):
    term_width = shutil.get_terminal_size((80, 20)).columns

    output_lines = [
        "╭───────────── Inference Stats ─────────────╮",
        f"│ Inference Time : {inf_time*1000:7.1f} ms",
        f"│ Post-Process   : {post_time*1000:7.1f} ms",
        f"│ Detections     : {num_boxes:7d}",
        f"│ Avg FPS        : {avg_fps:7.2f}",
        "╰────────────────────────────────────────────╯",
        "╭─────────────── Coverage % ───────────────╮",
    ]

    for label in camera_labels:
        output_lines.append(f"│ {label.capitalize():<6}: {coverage_dict[label]:.3f}")

    output_lines.append("╰────────────────────────────────────────────╯")

    output = "\n".join(line.ljust(term_width) for line in output_lines)
    print(output)
    clear_console_lines(len(output_lines))

# ───── Capture Thread ───── #
def capture_loop():
    global stitched_frame
    while not stop_event.is_set():
        frames = []
        for cam in cams:
            if cam:
                ret, frame = cam.read()
                if ret:
                    resized = cv2.resize(frame, (camera_width, camera_height))
                else:
                    resized = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            else:
                resized = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            frames.append(resized)
        top = cv2.hconcat([frames[0], frames[1]])
        bottom = cv2.hconcat([frames[2], frames[3]])
        stitched = cv2.vconcat([top, bottom])
        with frame_lock:
            stitched_frame = stitched
        time.sleep(0.005)

# ───── Inference Thread ───── #
def inference_loop():
    global inference_boxes, coverage_data
    while not stop_event.is_set():
        with frame_lock:
            if stitched_frame is None:
                frame_copy = None
            else:
                frame_copy = stitched_frame.copy()

        if frame_copy is None:
            time.sleep(0.002)
            continue

        t0 = time.perf_counter()
        results = model.predict(frame_copy, conf=conf_threshold, classes=[0], verbose=False)[0]
        t1 = time.perf_counter()

        boxes = results.boxes.data.cpu().numpy().tolist()
        camera_coverage = {side: 0.0 for side in camera_labels}
        for x1, y1, x2, y2, conf, cls in boxes:
            if conf < conf_threshold or int(cls) != 0:
                continue
            box_area = (x2 - x1) * (y2 - y1)
            col = int((x1 + x2) // 2 // camera_width)
            row = int((y1 + y2) // 2 // camera_height)
            index = row * grid_cols + col
            if 0 <= index < len(camera_labels):
                camera_coverage[camera_labels[index]] += box_area / (camera_width * camera_height)
        t2 = time.perf_counter()

        inference_duration = t1 - t0
        inference_times.append(inference_duration)
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        print_static_output(inference_duration, t2 - t1, len(boxes), avg_fps, camera_coverage)

        with inference_lock:
            inference_boxes = boxes
            coverage_data = camera_coverage

        try:
            tmp_path = json_output_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({k: round(v, 3) for k, v in camera_coverage.items()}, f)
            os.replace(tmp_path, json_output_path)
        except Exception as e:
            print(f"⚠️ JSON write error: {e}")

        time.sleep(0.002)

# ───── Display Function ───── #
def display_loop():
    if show_preview:
        cv2.namedWindow("YOLOv8 Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Preview", full_width, full_height)

    while not stop_event.is_set():
        with frame_lock:
            if stitched_frame is None:
                frame_disp = None
            else:
                frame_disp = stitched_frame.copy()
        if frame_disp is None:
            time.sleep(0.005)
            continue

        with inference_lock:
            boxes_copy = inference_boxes.copy() if inference_boxes else []
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
            avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        for x1, y1, x2, y2, conf, cls in boxes_copy:
            if conf < conf_threshold or int(cls) != 0:
                continue
            col = int((x1 + x2) // 2 // camera_width)
            row = int((y1 + y2) // 2 // camera_height)
            index = row * grid_cols + col
            label = camera_labels[index] if 0 <= index < len(camera_labels) else "unknown"
            cv2.rectangle(frame_disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_disp, label, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps_text = f"Avg Inference FPS: {avg_fps:.2f}"
        cv2.putText(frame_disp, fps_text, (10, full_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if show_preview:
            cv2.imshow("YOLOv8 Preview", frame_disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        else:
            time.sleep(0.005)

# ───── Start Threads ───── #
capture_thread = threading.Thread(target=capture_loop, daemon=True)
inference_thread = threading.Thread(target=inference_loop, daemon=True)

capture_thread.start()
inference_thread.start()

try:
    display_loop()
except KeyboardInterrupt:
    stop_event.set()

# ───── Cleanup ───── #
for cam in cams:
    if cam:
        cam.release()
if show_preview:
    cv2.destroyAllWindows()
print("Exited gracefully")
