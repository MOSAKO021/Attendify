import torch
import cv2
import numpy as np

# Load the pre-trained YOLOv5 model (YOLOv5s is the small version of the model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the image where you want to detect people
image_path = 'image.png'
image = cv2.imread(image_path)

# Run inference on the image
results = model(image)

# Extract detection information (bounding boxes, classes, etc.)
detections = results.pred[0]

# Filter detections to only count 'person' class (class ID for 'person' is 0)
people_detections = [det for det in detections if int(det[5]) == 0]  # Class 0 is 'person'

# Count the number of people detected
num_people = len(people_detections)

# Print out the result
print(f"Number of people detected: {num_people}")

# Display the image with bounding boxes around people
results.show()
