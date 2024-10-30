import os
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(input_path, invert_binary=False):
    logger.info(f"Processing image: {input_path}")
    
    # Check if file exists
    if not os.path.exists(input_path):
        logger.error(f"File not found: {input_path}")
        return None

    # Load image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Failed to load image: {input_path}")
        return None

    # Apply adaptive thresholding
    threshold_type = cv2.THRESH_BINARY_INV if invert_binary else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=threshold_type,
        blockSize=11,
        C=2
    )
    
    # Add border
    h, w = binary.shape
    bordered = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bordered[1:-1, 1:-1] = binary
    
    # Convert back to BGR for compatibility with the model
    bordered_bgr = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
    
    # Save preprocessed image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        cv2.imwrite(tmp_file.name, bordered_bgr)
        logger.info(f"Preprocessed image saved to: {tmp_file.name}")
        return tmp_file.name

def predict_with_preprocessing(model_path, image_path, conf_threshold=0.60, image_size=1920):
    # Load the model
    model = YOLO(model_path)
    
    # Preprocess the image
    preprocessed_image_path = preprocess_image(image_path)
    
    if preprocessed_image_path is None:
        logger.error("Preprocessing failed")
        return None
    
    try:
        # Run prediction on preprocessed image
        results = model.predict(
            source=preprocessed_image_path, 
            conf=conf_threshold, 
            imgsz=image_size
        )
        
        # Save results
        for result in results:
            result.save("output/output.jpg")
            
        logger.info("Prediction completed and saved")
        
        # Clean up temporary file
        os.unlink(preprocessed_image_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        if os.path.exists(preprocessed_image_path):
            os.unlink(preprocessed_image_path)
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 prediction with preprocessing')
    parser.add_argument('--model', required=True, help='Path to the YOLO model weights')
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--conf', type=float, default=0.60, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=1920, help='Image size')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run prediction with preprocessing
    results = predict_with_preprocessing(
        model_path=args.model,
        image_path=args.image,
        conf_threshold=args.conf,
        image_size=args.imgsz
    )
    
    if results is not None:
        logger.info("Processing completed successfully")
    else:
        logger.error("Processing failed")