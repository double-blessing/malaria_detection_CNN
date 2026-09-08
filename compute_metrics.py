import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import tensorflow as tf
from app import load_model, preprocess_image, CLASS_NAMES

# Ensure output directory exists
os.makedirs("results", exist_ok=True)

# 1. Load the fine-tuned model
print("[INFO] Loading fine-tuned model...")
model = load_model("mobilenetv2_finetuned")

# 2. Check if a test dataset folder is available
data_dir = "data/cell_images"  # Adjust to your local dataset path if present
if os.path.exists(data_dir):
    print(f"[INFO] Loading validation data from {data_dir}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(128, 128),
        batch_size=32,
    )

    y_true, y_pred_probs = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = (np.array(y_pred_probs) >= 0.5).astype(int)

    # Compute individual scores
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n--- Model Performance Metrics ---")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")

    # Detailed report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        cmap="Blues", ax=ax, values_format="d"
    )
    plt.title("Malaria Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    print("[INFO] Saved results/confusion_matrix.png")
else:
    print("[WARNING] Local 'data/cell_images' path not found.")
    print("[INFO] Inspect pre-generated matrices already stored in 'results/all_confusion_matrices.png'.")