# YOLO-CODE
A code that is used to detect protein crystals in images and draw bounding boxes around them.
This project involves using machine learning models to analyze images of protein crystals. 
This code is named YOLO code because it executes based on a YOLO model (YOLOv8). 
Its main function is detecting the protein crystals in the images and draw bounding boxes around them. 
It performs object detection and bounding box creation.

The code is displayed below.

!pip install ultralytics matplotlib opencv-python-headless
!pip install numpy

from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch

# Paths to dataset and files
dataset_path = "/content/drive/MyDrive/ENM401.v1-protein-crystals.yolov8-obb" **REPLACE WITH YOUR DATASET PATH**
data_yaml = os.path.join(dataset_path, "data.yaml")

# Ensure the paths exist
assert os.path.exists(dataset_path), "Dataset path does not exist!"
assert os.path.exists(data_yaml), "data.yaml file is missing!"

# Load YOLOv8 model (adjust model size if needed)
model = YOLO("yolov8n.pt")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify the Google Drive folder for saving training outputs
output_dir = "/content/drive/MyDrive/YOLOv8_Training_Output" **SELECT THE PATH FOR YOUR TRAINING OUTPUT**
os.makedirs(output_dir, exist_ok=True)

# Train the model
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=2,
    workers=2,
    cache=True,
    device=device,
    patience=10,
    project=output_dir,
    name="yolov8_protein_crystals"
)

print(f"Training outputs saved to: {output_dir}")

# Paths for inference
model_weights = os.path.join(output_dir, "yolov8_protein_crystals", "weights", "best.pt")
input_folder = "/content/drive/MyDrive/New" **SELECT THE INPUT FOLDER PATH FOR IMAGE INFERENCE**
output_folder = "/content/drive/MyDrive/YOLOv8_Output_Images" **SELECT THE OUTPUT FOLDER FOR IMAGE RESULTS**
os.makedirs(output_folder, exist_ok=True)

# Ensure paths exist
assert os.path.exists(model_weights), f"Model weights not found: {model_weights}"
assert os.path.exists(input_folder), f"Input folder not found: {input_folder}"

# Load the trained model
model = YOLO(model_weights)

def preprocess_image_clahe(image):
    """
    Apply CLAHE and sharpening techniques to enhance image quality.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(cv2.merge([equalized, equalized, equalized]), (5, 5), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(blurred, -1, kernel)

def adaptive_gamma_correction(image, gamma=1.2):
    """
    Perform adaptive gamma correction to improve contrast.
    """
    normalized = image / 255.0
    return np.uint8(np.power(normalized, gamma) * 255)

def preprocess_image(image):
    """
    Combine preprocessing methods for enhanced image quality.
    """
    return adaptive_gamma_correction(preprocess_image_clahe(image), gamma=1.2)

def draw_bounding_boxes(image, boxes, classes, confidences, class_colors):
    """
    Draw bounding boxes on the image with larger, bold font size and thicker boxes.
    """
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        cls_adjusted = int(cls) + 1  # Adjust class index to start from 1
        label = f"{cls_adjusted} {conf:.2f}"
        color = class_colors[cls_adjusted] if cls_adjusted in class_colors else (255, 255, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)  # Thicker bounding box
        # Draw label with bold font
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # Larger, bolder font
    return image

def run_inference_with_preprocessing(model, input_folder, output_folder):
    """
    Run inference on all images in a folder with preprocessing.

    Args:
        model: Loaded YOLO model.
        input_folder: Path to folder with input images.
        output_folder: Path to folder for saving results.
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    class_colors = {
        1: (0, 0, 255),  # Blue for Class 1
        2: (255, 0, 0),  # Red for Class 2
        3: (0, 255, 0),  # Green for Class 3
    }

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(valid_extensions):
            image_path = os.path.join(input_folder, file_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipping file {file_name}: unable to read image.")
                continue

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Run YOLO inference
            results = model(preprocessed_image)
            detections = results[0].boxes
            boxes = detections.xyxy.cpu().numpy()
            confidences = detections.conf.cpu().numpy()
            classes = detections.cls.cpu().numpy()

            # Draw bounding boxes
            annotated_image = draw_bounding_boxes(preprocessed_image, boxes, classes, confidences, class_colors)

            # Save the annotated image
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved at: {output_path}")

# Run inference
run_inference_with_preprocessing(model, input_folder, output_folder)

print(f"All results saved in: {output_folder}")
