import json
import os
import re

import cv2
import easyocr

from utils import preprocess_image


def extract_info(image_path, output_dir):
    """Extract information from NID image using OCR."""
    # Preprocess image
    preprocessed = preprocess_image(image_path)
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(preprocessed)
    
    # Parse extracted text
    extracted_info = {
        "Full Name": "",
        "NID Number": "",
        "Date of Birth": "",
        "Address": ""
    }
    
    name_pattern = re.compile(r"Name[:\s]*(.+)", re.IGNORECASE)
    nid_pattern = re.compile(r"NID[:\s]*(\d+)", re.IGNORECASE)
    dob_pattern = re.compile(r"DOB[:\s]*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
    address_pattern = re.compile(r"Address[:\s]*(.+)", re.IGNORECASE)
    
    for detection in result:
        text = detection[1]
        if name_match := name_pattern.search(text):
            extracted_info["Full Name"] = name_match.group(1).strip()
        if nid_match := nid_pattern.search(text):
            extracted_info["NID Number"] = nid_match.group(1).strip()
        if dob_match := dob_pattern.search(text):
            extracted_info["Date of Birth"] = dob_match.group(1).strip()
        if address_match := address_pattern.search(text):
            extracted_info["Address"] = address_match.group(1).strip()
    
    # Save to JSON
    output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}.json")
    with open(output_path, "w") as f:
        json.dump(extracted_info, f, indent=4)
    
    return extracted_info

# Example usage
if __name__ == "__main__":
    cropped_image = "output/detections/crops/NID/sample.jpg"
    extract_info(cropped_image, "output/extracted_info")