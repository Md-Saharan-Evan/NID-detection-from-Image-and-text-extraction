import json
import re

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO


def preprocess_image_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned


def extract_text_info(text):
    """Extract Full Name, NID Number, DOB, Address from raw OCR text."""
    result = {
        "Full Name": None,
        "NID Number": None,
        "Date of Birth": None,
        "Address": None,
        "Raw Text": text
    }

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        if len(line) > 4 and all(c.isalpha() or c.isspace() for c in line):
            result["Full Name"] = line
            break
    nid_match = re.search(r'\b\d{10,17}\b', text)
    if nid_match:
        result["NID Number"] = nid_match.group(0)

    dob_match = re.search(r'\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4}', text)
    if dob_match:
        result["Date of Birth"] = dob_match.group(0)

    keywords = ['road', 'village', 'district', 'post', 'p.o', 'thana', 'area']
    address_lines = [line for line in lines if any(kw in line.lower() for kw in keywords)]
    if address_lines:
        result["Address"] = " ".join(address_lines)

    return result


def run_pipeline(image_path, yolo_model_path, debug=False):
    model = YOLO(yolo_model_path)
    results = model(image_path)
    image = cv2.imread(image_path)

    if len(results[0].boxes) == 0:
        return {"error": "No NID detected"}

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = results[0].names[int(box.cls[0])] if hasattr(results[0], 'names') else "NID"

        # Draw bounding box on image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Crop and preprocess for OCR
        cropped_nid = image[y1:y2, x1:x2]
        preprocessed = preprocess_image_for_ocr(cropped_nid)

        # Debug view
        if debug:
            cv2.imshow("Preprocessed OCR Input", preprocessed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("debug_preprocessed.jpg", preprocessed)

        # OCR
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:/.- '
        text = pytesseract.image_to_string(preprocessed, config=config)

        extracted_info = extract_text_info(text)

        # Save output image
        cv2.imwrite("output_with_bbox.jpg", image)

        return extracted_info

    return {"error": "Failed to process image"}
if __name__ == "__main__":
    image_path = "Testing_data/00003620_in.jpg"
    yolo_model_path = "runs/detect/train5/weights/best.pt"

    result = run_pipeline(image_path, yolo_model_path, debug=True)
    print(json.dumps(result, indent=4))
