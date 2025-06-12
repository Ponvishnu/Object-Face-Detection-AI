# 🧠 Object & Face Detection AI

A computer vision system that detects human faces and real-world objects using OpenCV and deep learning. It includes separate modules for image-based and real-time webcam detection using YOLOv3 and Haar Cascades.

## 🧠 Features
- 🧍 Detect human faces using Haar Cascade classifiers
- 🎯 Detect 80+ objects using YOLOv3 (cars, people, bikes, etc.)
- 📷 Supports both image and webcam input
- 📦 Draws bounding boxes with confidence scores
- 🖼️ Real-time detection with live webcam feed

# 🚀 How to Run
## 📸 For Image/Object Detection:
python detect_objects.py
### 🙂 For Face Detection:
python detect_faces.py
### 🎥 For Real-Time Webcam Detection:
python webcam_detector.py
Ensure all model files are in the models/ directory before running.

## 📦 Model Downloads
🔗 YOLOv3 Weights
🔗 YOLOv3 Config (cfg)
🔗 COCO Labels
Place these files in the models/ folder.
 
## 🧪 Requirements
Python 3.10+
opencv-python
numpy
Install dependencies:
pip install opencv-python numpy
📜 This project is licensed under the MIT License.
