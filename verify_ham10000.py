import os
import sys
import csv
import glob
import time
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from api.services.inference import get_model
from src.config import DATASET

# Assuming the directory passed contains images named with their class as prefix or part of name, 
# or a ground truth CSV exists. For this utility, we will try to infer ground truth from filename 
# (e.g. ISIC_00000_MEL.jpg) OR just leave ground truth empty if unknown, but the prompt says 
# "Known HAM10000 validation images". Typically HAM10000 uses a metadata CSV.

def main(image_dir):
    model = get_model()
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not image_paths:
        print(f"No JPG images found in {image_dir}")
        return

    # Try to find metadata.csv in the directory or parent
    metadata_path = os.path.join(image_dir, "HAM10000_metadata.csv")
    gt_dict = {}
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        for _, row in df.iterrows():
            gt_dict[row['image_id']] = row['dx'].upper() # e.g., 'mel', 'bcc'
    
    results = []
    print(f"Running batch inference on {len(image_paths)} images...")
    
    for path in image_paths:
        filename = os.path.basename(path)
        image_id = filename.split('.')[0]
        
        # Ground truth
        gt = gt_dict.get(image_id, "UNKNOWN")
        if gt == "UNKNOWN":
            # Attempt to extract from filename if formatted like ISIC_00000_MEL.jpg
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) > 1 and parts[-1].upper() in DATASET.class_names:
                gt = parts[-1].upper()
                
        # Inference
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            x = np.expand_dims(img, axis=0)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x.astype(np.float32))
            
            preds = model.predict(x, verbose=0)[0]
            top_idx = np.argmax(preds)
            prediction = DATASET.class_names[top_idx].upper()
            confidence = float(preds[top_idx])
            
            correct = "YES" if gt != "UNKNOWN" and gt == prediction else ("NO" if gt != "UNKNOWN" else "N/A")
            
            results.append([filename, gt, prediction, f"{confidence:.4f}", correct])
            print(f"Processed {filename} -> GT: {gt}, Pred: {prediction} ({correct})")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    out_csv = "verification_predictions.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Ground Truth", "Prediction", "Confidence", "Correct/Incorrect"])
        writer.writerows(results)
    
    print(f"\nVerification complete. Results saved to {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_ham10000.py <path_to_images_directory>")
        sys.exit(1)
    main(sys.argv[1])
