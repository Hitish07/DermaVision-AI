# FINAL RESEARCH RESULTS: DermaVision AI

## 1. Threats to Validity (Data Leakage Advisory)
**IMPORTANT ACADEMIC CONSTRAINT**: The original `train/val/test` split metadata (including the initial random seed) was unavailable during this evaluation phase. To prevent data leakage during this benchmark, a pseudo-test split was reconstructed utilizing `GroupShuffleSplit` on the `lesion_id` to strictly prevent patient overlap. 
However, because the original random seed is unknown, the model may have seen these specific images during its original training. Therefore, the reported metrics represent a **Reproducibility Evaluation** rather than an absolute independent benchmark. The evaluation protocol remains mathematically sound, but is not identical to the original experiment.

## 2. Experimental Setup & Hardware Profiling
- **Architecture**: MobileNetV2 (Pretrained on ImageNet, Fine-tuned)
- **Total Parameters**: 3,050,055
- **Model Load Time**: 1.03 seconds
- **Average Inference Latency**: 150.68 ms per image
- **Evaluation Set Size**: 100 images (Randomly sampled from pseudo-test reconstruct)

## 3. Quantitative Results
- **Accuracy**: 0.4800
- **Macro Precision**: 0.1359
- **Macro Recall**: 0.2823
- **Macro F1 Score**: 0.1370
- **ROC-AUC (Macro)**: 0.7409
- **Matthews Correlation Coefficient (MCC)**: 0.1318
- **Cohen's Kappa**: 0.1212
- **Log Loss**: 2.0521

## 4. Class-wise Analysis & Failure Analysis
Class-specific F1 and Precision distributions are available in `class_performance.csv`.
The highest-confidence false positives have been exported to `failure_analysis.csv`. Analysis shows that standard domain shift (e.g., non-macro imagery) heavily impacts the confidence distributions, resulting in over-confident misclassifications of background artifacts.

## 5. Publication Figures
Publication quality (300 DPI) matrices are available in the `reports/academic_evaluation/figures/` directory:
- `confusion_matrix.png`
- `roc_curve.png`

## 6. Limitations & Future Work
- Due to the aforementioned `Threats to Validity`, a complete retraining from scratch using a cryptographically saved seed is required for submission to peer-reviewed clinical journals.
- The system heavily relies on dermatoscopic lenses; future work should explore domain-adaptation techniques (e.g. CycleGAN) to map smartphone images into the dermatoscopic distribution prior to inference.
