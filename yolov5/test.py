import torch
from PIL import Image

# Load your saved model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./runs/train/exp8/weights/last.pt', force_reload=True)

def detect_humans(image_path, output_path):
    # Load the image
    img = Image.open(image_path)
    
    # Run the model on the image
    results = model(img)
    
    # Filter results for 'person' class (assuming class '0' is human)
    detected_humans = results.pred[0][results.pred[0][:, -1] == 0]  # Replace '0' with the correct class ID for humans if necessary
    
    # Print the count of detected humans
    print(f"Total number of humans detected: {len(detected_humans)}")
    
    # Render results on the image and save it
    results.save(save_dir='./outp/output/')  # This will save the output in the same directory by default
    
    # # Rename and move the saved output image
    # output_img = Image.open("./outp/output/image1.jpg")  # YOLOv5 saves with default name 'image0.jpg'
    # output_img.save(output_path)

# Usage
input_image = 'image2.png'     # Input image path
output_image = 'output.jpg'   # Output image path
detect_humans(input_image, output_image)
