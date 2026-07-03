# Project Architecture

The DermaVision AI ecosystem is partitioned into four distinct layers:

1. **Presentation Layer (React 18 + Vite)**
   Handles client interaction, asynchronous polling for XAI tasks, and visual rendering of heatmaps.

2. **API & Orchestration Layer (FastAPI)**
   Handles RESTful ingress, OpenCV quality assessments, MobileNetV2 heuristic validation, and SQLite persistence.

3. **Core AI Inference Engine (TensorFlow/Keras)**
   Modularized under `src/`. Executes Hair Removal (DullRazor approximation), CLAHE, and forwards through the primary diagnostic CNN.

4. **Explainability Engine (XAI)**
   Runs parallel execution paths. Grad-CAM executes synchronously. Integrated Gradients and SHAP utilize `BackgroundTasks` to prevent HTTP thread blocking.
