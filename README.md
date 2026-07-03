# DermaVision AI

A professional real-world Clinical Decision Support Platform for Skin Lesion Analysis, built with TensorFlow, FastAPI, and React.

## Clinical Multi-Stage Pipeline

DermaVision AI utilizes a robust 4-stage pipeline to ensure clinical integrity:

1. **Intelligent Image Validation**: Uses a combination of YCrCb color-space segmentation and a MobileNetV2 blocklist to reject images that do not contain human skin (e.g., dogs, cats, furniture). It also detects whether an image is dermoscopic or a standard smartphone photo.
2. **Quality Assessment**: Computes blur (Laplacian Variance), brightness, contrast, and noise to reject unreadable images.
3. **Morphological Lesion Detection**: Uses DullRazor hair removal, CLAHE enhancement, and Adaptive Thresholding to isolate the skin lesion contour, proving that a morphological anomaly exists prior to inference.
4. **Deep Learning Inference**: A custom-trained MobileNetV2 model classifies the lesion into one of 7 HAM10000 disease categories. Predictions are automatically mapped to clinical risk profiles (High, Medium, Low).
5. **Explainable AI (XAI)**: Generates Grad-CAM, SHAP, and Integrated Gradients overlays to provide transparent visual evidence for the model's decision.

## Architecture

- **Backend**: FastAPI (Python), SQLite
- **Frontend**: React, Tailwind CSS, Framer Motion
- **AI/ML Engine**: TensorFlow, OpenCV, scikit-learn

## Limitations & Disclaimers

**MEDICAL DISCLAIMER**: DermaVision AI is an experimental clinical decision-support tool designed for educational and research purposes. It is **NOT** FDA approved and should **never** be used as a substitute for professional medical diagnosis.

* **Dataset Bias**: The model is trained strictly on the HAM10000 dataset, which consists primarily of dermoscopic images. Predictions made on ordinary smartphone photographs have significantly reduced reliability. 
* **Heuristic Limitations**: The lesion detection pipeline utilizes classical computer vision (OpenCV). While highly effective, it may struggle with extreme edge cases (e.g., heavy shadows, extreme hair). Future upgrades may replace this with a dedicated YOLOv8 segmentation model.
