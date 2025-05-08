NID-detection-from-Image-and-text-extraction
This project performs(NID) detection from an image using YOLOv8 and extracts key information such as Name, NID Number, Date of Birth, and Address using OCR (Tesseract).


1. Create a Virtual Environment
python -m venv env
source env/bin/activate

2. Install Required Dependencies
pip install -r requirements.txt


3. Project Structure

Data folder contails 2 folder 1 for images and another is labels

4. create the dataset.yaml file 


5. Training the YOLOv8 Model
Use the ultralytics CLI or Python API:

yolo detect train data=data/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640

6. we can see the training and testing performance along with accuracy metrics in runs/detect/train5
   Accuracy metrics used metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
   All the confusion metrix curves and other along with results are availe in the train5 folder


8. python info_extract.py
Update image_path and yolo_model_path in the script to match your test image and YOLOv8 weight file.

Output Example:

{
    "Full Name": "MD SAHARAN EVAN",
    "NID Number": "1992387483748",
    "Date of Birth": "12/05/1998",
    "Address": "Village: X, Post: Y, District: Z",
    "Raw Text": "..."
}
Testing YOLO with Bounding Box and Text
To visualize bounding boxes with class labels:


Model Evaluation using mAP, Precision, Recall (via YOLOv8 metrics)
