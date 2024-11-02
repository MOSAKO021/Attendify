import torch
import cv2
import numpy as np
import urllib.request

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained YOLOv5 model (YOLOv5s is the small version of the model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)  # Load model to CUDA if available

# Load the image from a URL (you cannot use cv2.imread directly for URLs)
url = 'https://teleuniv.net.in/cctv/cctv1.php?id=21'
resp = urllib.request.urlopen(url)
image_array = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Run inference on the image (move the input image to the same device as the model)
results = model(image)

# Extract detection information (bounding boxes, classes, etc.)
detections = results.pred[0].to(device)  # Ensure detections are also moved to the GPU/CPU

# Filter detections to only count 'person' class (class ID for 'person' is 0)
people_detections = [det for det in detections if int(det[5]) == 0]  # Class 0 is 'person'

# Count the number of people detected
num_people = len(people_detections)

# Print out the result
print(f"Number of people detected: {num_people}")

# Draw bounding boxes only around detected people
for det in people_detections:
    x1, y1, x2, y2, conf, class_id = det
    # Draw bounding box on the original image (in green with thickness 2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # Add label 'person' on top of the bounding box
    cv2.putText(image, 'person', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Create a named window for the display and resize the window to a smaller size (e.g., 640x480)
cv2.namedWindow("People Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("People Detection", 640, 480)  # Set the window size to 640x480

# Display the image with bounding boxes around people
cv2.imshow("People Detection", image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
