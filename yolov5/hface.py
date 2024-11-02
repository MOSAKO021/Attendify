import torch
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)

# Enable FP16 for faster processing if using a GPU
print(device.type)
if device.type == 'cuda':
    model.half()

# Replace '0' with the URL from the IP Webcam app
url = "http://192.168.24.13:8080/video"
cap = cv2.VideoCapture(url)

# Create a named window for the camera feed and set it to full screen
cv2.namedWindow('Real-Time Object Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Real-Time Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Frame processing control variables
frame_count = 0
frame_skip = 5  # Process every 5th frame to reduce lag

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames to reduce processing load
    if frame_count % frame_skip != 0:
        continue

    # Resize the frame to reduce processing load (e.g., 1280x720)
    frame = cv2.resize(frame, (1280, 720))

    # Convert the frame to a PIL Image for the model (DETR expects PIL Images)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prepare inputs for the model and send them to the appropriate device (GPU/CPU)
    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    # Convert the input tensors to half precision if using a GPU
    if device.type == 'cuda':
        inputs['pixel_values'] = inputs['pixel_values'].half()

    # Model inference
    outputs = model(**inputs)

    # Synchronize GPU operations to avoid lag
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Get the image dimensions
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)  # Width and height

    # Post-process the model outputs to get the bounding boxes and labels
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]

    # Counter for the number of persons detected
    person_count = 0

    # Loop through detected objects and draw bounding boxes for persons only
    for _, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() == 1:  # '1' is the label ID for "person" in the COCO dataset
            person_count += 1  # Increment the person count
            box = [round(i) for i in box.tolist()]
            label_name = model.config.id2label[label.item()]

            # Draw bounding boxes (in green) and the label text on the frame
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'{label_name}', (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the total number of persons detected in the frame
    cv2.putText(frame, f'Total Persons: {person_count}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Display the frame with the detections and person count
    cv2.imshow('Real-Time Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video feed and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
