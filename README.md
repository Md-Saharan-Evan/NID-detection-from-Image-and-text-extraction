NID-detection-from-Image-and-text-extraction
This project performs(NID) detection from an image using YOLOv8 and extracts key information such as Name, NID Number, Date of Birth, and Address using OCR (Tesseract).

🔧 Setup Instructions
1. Create a Virtual Environment
bash
Copy
Edit
python -m venv env
source env/bin/activate  
2. Install Required Dependencies
bash
Copy
Edit
pip install -r requirements.txt


📁 Project Structure
bash
Copy
Edit
.
├── data/
│   ├── images/         # Training/test images
│   ├── labels/         # YOLO-format annotations
│   └── dataset.yaml    # YOLOv8 dataset config
├── runs/               # YOLO training results
├── Testing_data/       # Sample test images
├── src/
│   └── dataset.py      # Augmentation + Dataset split script
├── info_extract.py     # Main pipeline (detection + OCR + JSON)
├── requirements.txt
└── README.md
🏗️ Training the YOLOv8 Model
Use the ultralytics CLI or Python API:

bash
Copy
Edit
yolo detect train data=data/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
Make sure your image-label pairs are organized as:

kotlin
Copy
Edit
data/images/train/
data/labels/train/
data/images/val/
data/labels/val/
🔍 Run Detection + Text Extraction
Once the model is trained, run the full detection-OCR pipeline:

bash
Copy
Edit
python info_extract.py
Update image_path and yolo_model_path in the script to match your test image and YOLOv8 weight file.

✅ Output Example
json
Copy
Edit
{
    "Full Name": "MD SAHARAN EVAN",
    "NID Number": "1992387483748",
    "Date of Birth": "12/05/1998",
    "Address": "Village: X, Post: Y, District: Z",
    "Raw Text": "..."
}
🧪 Testing YOLO with Bounding Box and Text
To visualize bounding boxes with class labels:

python
Copy
Edit
results = model(image_path)
results[0].plot()
You can also draw boxes manually in cv2 using:

python
Copy
Edit
cv2.rectangle(...)
cv2.putText(...)
🛠️ Improvements Used
Data Augmentation with Albumentations

OCR Preprocessing: Denoising, Thresholding, Resizing

Regex-based parsing for structured data

Model Evaluation using mAP, Precision, Recall (via YOLOv8 metrics)
