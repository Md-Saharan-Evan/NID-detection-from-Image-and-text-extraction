import cv2
import numpy as np


def preprocess_image(image_path):
    """Preprocess image for OCR."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Denoise
    img = cv2.fastNlMeansDenoising(img)
    # Adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    # Save preprocessed image temporarily
    temp_path = "temp_preprocessed.jpg"
    cv2.imwrite(temp_path, img)
    return temp_path