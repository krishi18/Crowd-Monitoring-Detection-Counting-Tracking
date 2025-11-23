import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker


# video_path = os.path.join('.', 'data', 'people.mp4')
video_path = r"C:\Users\Gabriel\output_video.mp4"
video_out_path = os.path.join('.', 'retarded.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MPV4'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# model = YOLO(r"C:\Users\Gabriel\Downloads\yoloFinal.pt")
model = YOLO("yolov8m.pt")
# model = YOLO(r"C:\Users\Gabriel\Downloads\RTDETRFinal.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
            # class_name = model.names.get(class_id, "Unknown")  # Get class name from model
            # print(f"Detected class: {class_name} (ID: {class_id})")


        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()













# # ----------------------------------------------------------------------------------------------------------------------------------------------
# import os
# import random
# import cv2
# from ultralytics import YOLO
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction
# from tracker import Tracker
# import numpy as np

# # Video paths
# video_path = r"C:\Users\Gabriel\output_video.mp4"
# video_out_path = os.path.join('.', 'outmotsahi.mp4')

# # Open video capture
# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()

# # Initialize video writer
# cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
#                           (frame.shape[1], frame.shape[0]))

# # Initialize the tracker
# tracker = Tracker()

# # Generate random colors for tracking boxes
# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# # Detection threshold
# detection_threshold = 0.5

# # Initialize the SAHI detection model for YOLOv8
# detection_model = AutoDetectionModel.from_pretrained(
#     model_type="yolov8",
#     model_path="yolov8m.pt",  # Replace with your model path
#     confidence_threshold=detection_threshold,
#     device="cpu"  # Change to 'cuda:0' if using a GPU
# )

# # Start processing the video frames
# while ret:
#     # Get predictions with SAHI (slicing the frame)
#     prediction = get_sliced_prediction(
#         frame,  # Frame as numpy array
#         detection_model,
#         slice_height=256,  # Slice height for splitting the frame into smaller parts
#         slice_width=256,  # Slice width for splitting the frame into smaller parts
#         overlap_height_ratio=0.2,  # Overlap ratio for vertical slices
#         overlap_width_ratio=0.2,  # Overlap ratio for horizontal slices
#     )

#     detections = []
#     for detection in prediction.object_prediction_list:  # Access detections from result
#         # Assuming the detection object has a 'bbox' attribute in the format [x, y, width, height]
#         bbox = detection.bbox  # `bbox` is a BoundingBox object

#         # Access the coordinates of the bounding box
#         x1 = bbox.minx
#         y1 = bbox.miny
#         x2 = bbox.maxx
#         y2 = bbox.maxy

#         # Calculate width and height
#         width = x2 - x1
#         height = y2 - y1

#         score = detection.score
#         category_id = detection.category_id
#         # Convert bbox to corner coordinates if needed
#         x1 = x
#         y1 = y
#         x2 = x + width
#         y2 = y + height

#         if score > detection_threshold:
#             detections.append([x1, y1, x2, y2, score])

#     # Update the tracker with the detected bounding boxes
#     tracker.update(frame, detections)

#     # Draw tracking boxes on the frame
#     for track in tracker.tracks:
#         bbox = track.bbox
#         x1, y1, x2, y2 = bbox
#         track_id = track.track_id
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

#     # Write the processed frame to the output video
#     cap_out.write(frame)

#     # Read the next frame
#     ret, frame = cap.read()

# # Release video objects
# cap.release()
# cap_out.release()
# cv2.destroyAllWindows()
