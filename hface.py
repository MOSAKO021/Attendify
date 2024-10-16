# from transformers import DetrImageProcessor, DetrForObjectDetection
# import torch
# from PIL import Image
# import requests

# url = "./image1.png"
# # image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open(url)

# # you can specify the revision tag if you don't want the timm dependency
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)

# # convert outputs (bounding boxes and class logits) to COCO API
# # let's only keep detections with score > 0.9
# target_sizes = torch.tensor([image.size[::-1]])
# results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#             f"Detected {model.config.id2label[label.item()]} with confidence "
#             f"{round(score.item(), 3)} at location {box}"
#     )





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

# Replace '0' with the URL from the IP Webcam app
url = "http://192.168.1.5:8080/video"
cap = cv2.VideoCapture(url)

# Create a named window for the camera feed and set it to full screen
cv2.namedWindow('Real-Time Object Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Real-Time Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to a PIL Image for the model (DETR expects PIL Images)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prepare inputs for the model and send them to the appropriate device (GPU/CPU)
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Get the image dimensions
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)  # Width and height

    # Post-process the model outputs to get the bounding boxes and labels
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Loop through detected objects and draw bounding boxes on the frame
    for _, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]

        # Draw bounding boxes (in green) and the label text on the frame
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, f'{label_name}', (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the detections
    cv2.imshow('Real-Time Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video feed and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
