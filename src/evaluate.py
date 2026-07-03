import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, accuracy_score
import tensorflow as tf
import logging
import time
from .config import PATHS, DATASET

logger = logging.getLogger(__name__)

def evaluate_model(model, model_name, test_dataset, y_true):
    """Generates all evaluation metrics and plots for a model."""
    logger.info(f"Evaluating {model_name}...")
    
    start_time = time.time()
    predictions = model.predict(test_dataset)
    inference_time = time.time() - start_time
    
    y_pred = np.argmax(predictions, axis=1)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=DATASET.class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(PATHS.reports / f"{model_name}_classification_report.csv")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=DATASET.class_names, yticklabels=DATASET.class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(PATHS.figures / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Advanced Metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Macro_F1': report['macro avg']['f1-score'],
        'MCC': mcc,
        'Cohen_Kappa': kappa,
        'Inference_Time_s': inference_time,
        'Parameters': model.count_params()
    }
    
    pd.DataFrame([metrics]).to_csv(PATHS.reports / f"{model_name}_metrics.csv", index=False)
    
    # Plot ROC and PR Curves
    plot_roc_curves(y_true, predictions, model_name)
    plot_pr_curves(y_true, predictions, model_name)
    
    return metrics

def plot_roc_curves(y_true, y_pred_proba, model_name):
    y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=DATASET.num_classes)
    plt.figure(figsize=(10, 8))
    
    for i in range(DATASET.num_classes):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{DATASET.class_names[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(PATHS.figures / f"{model_name}_roc_curve.png", dpi=300)
    plt.close()

def plot_pr_curves(y_true, y_pred_proba, model_name):
    y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=DATASET.num_classes)
    plt.figure(figsize=(10, 8))
    
    for i in range(DATASET.num_classes):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{DATASET.class_names[i]} (AUC = {pr_auc:.2f})')
        
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(PATHS.figures / f"{model_name}_pr_curve.png", dpi=300)
    plt.close()
