"""Evaluate the locally available models using their training preprocessing."""

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "cell_images"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_NAMES = ["Parasitized", "Uninfected"]
VALIDATION_SPLIT = 0.2


def build_custom_cnn():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="Custom_CNN",
    )


def build_mobilenetv2_frozen():
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(128, 128, 3)
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(base.input, output, name="MobileNetV2_Frozen")


def make_generator(preprocessing):
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")
    missing = [name for name in CLASS_NAMES if not (DATA_DIR / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Expected class folders under {DATA_DIR}; missing: {', '.join(missing)}"
        )

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing,
        validation_split=VALIDATION_SPLIT,
    ).flow_from_directory(
        str(DATA_DIR),
        classes=CLASS_NAMES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )
    expected_mapping = {"Parasitized": 0, "Uninfected": 1}
    print("\nClass mapping:")
    print(generator.class_indices)
    if generator.class_indices != expected_mapping:
        raise ValueError(f"Unexpected class mapping: {generator.class_indices}")
    return generator


def save_confusion_matrix(cm, model_key):
    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="Actual",
        title=f"{model_key} confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.tight_layout()
    output = RESULT_DIR / f"{model_key}_confusion_matrix.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output.relative_to(ROOT)).replace("\\", "/")


def evaluate_model(model, weight_path, model_key, display_name, generator):
    if not weight_path.is_file():
        return None

    print(f"\nLoading {display_name}: {weight_path.name}")
    model.load_weights(weight_path)
    model(generator[0][0], training=False)  # Build/warm up before timing.

    probabilities_uninfected = []
    elapsed = 0.0
    for batch_index in range(len(generator)):
        images, _ = generator[batch_index]
        start = time.perf_counter()
        batch_output = model(images, training=False).numpy().reshape(-1)
        elapsed += time.perf_counter() - start
        probabilities_uninfected.extend(batch_output.tolist())

    # The sigmoid is trained against directory labels: 0=Parasitized, 1=Uninfected.
    probabilities_uninfected = np.asarray(probabilities_uninfected, dtype=np.float64)
    probabilities_parasitized = 1.0 - probabilities_uninfected
    y_true = np.asarray(generator.classes, dtype=np.int32)
    y_pred = np.where(probabilities_parasitized >= 0.5, 0, 1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp_para, fn_para = int(cm[0, 0]), int(cm[0, 1])
    fp_para, tn_para = int(cm[1, 0]), int(cm[1, 1])

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    precision = ratio(tp_para, tp_para + fp_para)
    recall = ratio(tp_para, tp_para + fn_para)
    specificity = ratio(tn_para, tn_para + fp_para)
    f1 = ratio(2 * precision * recall, precision + recall)
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": float(roc_auc_score(y_true == 0, probabilities_parasitized)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "parameters": int(model.count_params()),
        "inference_time_seconds": float(elapsed),
        "inference_time_ms_per_image": float(1000 * elapsed / len(y_true)),
        "evaluated_images": int(len(y_true)),
        "preprocessing": (
            "RGB float pixel values divided by 255" if model_key == "custom_cnn"
            else "keras.applications.mobilenet_v2.preprocess_input (0-255 RGB to -1..1)"
        ),
        "confusion_matrix_file": save_confusion_matrix(cm, model_key),
    }
    return metrics


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    available = {
        "custom_cnn": (
            "Custom CNN",
            MODEL_DIR / "custom_cnn_weights_only.weights.h5",
            build_custom_cnn,
            lambda image: image / 255.0,
        ),
        "mobilenetv2_frozen": (
            "MobileNetV2 Frozen",
            MODEL_DIR / "mobilenetv2_frozen_weights_only.weights.h5",
            build_mobilenetv2_frozen,
            tf.keras.applications.mobilenet_v2.preprocess_input,
        ),
    }
    present = [key for key, item in available.items() if item[1].is_file()]
    if not present:
        raise FileNotFoundError(f"No supported weights found in {MODEL_DIR}")

    results = {}
    for key in present:
        display_name, _, _, preprocess = available[key]
        print(f"\nPreparing validation images for {display_name}")
        generator = make_generator(preprocess)
        _, weight_path, builder, _ = available[key]
        metrics = evaluate_model(builder(), weight_path, key, display_name, generator)
        if metrics is not None:
            results[display_name] = metrics

    output = {
        "dataset": str(DATA_DIR.relative_to(ROOT)).replace("\\", "/"),
        "evaluation_split": "20% validation subset, matching notebook ImageDataGenerator split",
        "class_mapping": {"Parasitized": 0, "Uninfected": 1},
        "images_found": int(generator.samples),
        "models": results,
    }
    output_path = RESULT_DIR / "model_evaluation_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved measured metrics to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
