"""Run Grad-CAM smoke checks on one real image from each dataset class.

Run from the project root with:
    .\.venv\Scripts\python.exe test_gradcam.py
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.gradcam import find_last_conv_layer, generate_gradcam_heatmap, overlay_gradcam


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "cell_images"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "results" / "gradcam"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_custom_cnn():
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="custom_conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="custom_conv2")(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="Custom_CNN")


def build_mobilenetv2_frozen():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(base.input, outputs, name="MobileNetV2_Frozen")


def preprocess(image, model_name):
    array = np.asarray(image.resize((128, 128)), dtype=np.float32)
    if model_name == "MobileNetV2 Frozen":
        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    else:
        array /= 255.0
    return np.expand_dims(array, axis=0)


def first_readable_image(class_name):
    folder = DATA_DIR / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset class folder: {folder}")
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    return path, image.convert("RGB")
            except (OSError, ValueError):
                continue
    raise FileNotFoundError(f"No readable image found in {folder}")


def main():
    models = {
        "Custom CNN": (
            build_custom_cnn(),
            MODEL_DIR / "custom_cnn_weights_only.weights.h5",
        ),
        "MobileNetV2 Frozen": (
            build_mobilenetv2_frozen(),
            MODEL_DIR / "mobilenetv2_frozen_weights_only.weights.h5",
        ),
    }
    for model_name, (model, weights) in models.items():
        if not weights.is_file():
            raise FileNotFoundError(f"Missing model weights: {weights}")
        model.load_weights(weights)
        layer_name = find_last_conv_layer(model)
        print(f"{model_name} Grad-CAM layer: {layer_name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in ("Parasitized", "Uninfected"):
        sample_path, original = first_readable_image(class_name)
        slug = class_name.lower()
        print(f"\nInput sample ({class_name}): {sample_path.relative_to(ROOT)}")
        for model_name, (model, _) in models.items():
            tensor = preprocess(original, model_name)
            probability_uninfected = float(model(tensor, training=False)[0, 0])
            probabilities = {"Parasitized": 1.0 - probability_uninfected,
                             "Uninfected": probability_uninfected}
            prediction = max(probabilities, key=probabilities.get)
            confidence = probabilities[prediction]
            class_index = 0 if prediction == "Parasitized" else 1
            layer_name = find_last_conv_layer(model)
            heatmap = generate_gradcam_heatmap(
                model, tensor, layer_name, class_index=class_index
            )
            if heatmap.size == 0 or not np.isfinite(heatmap).all():
                raise AssertionError(f"Invalid heatmap generated for {sample_path}")
            overlay = overlay_gradcam(original, heatmap, alpha=0.4)
            model_slug = "custom_cnn" if model_name == "Custom CNN" else "mobilenetv2_frozen"
            if model_name == "Custom CNN":
                output_path = OUTPUT_DIR / f"{slug}_gradcam.png"
            else:
                output_path = OUTPUT_DIR / f"{slug}_{model_slug}_gradcam.png"
            overlay.save(output_path)
            print(
                f"{model_name}: prediction={prediction}, confidence={confidence:.2%}, "
                f"heatmap={heatmap.shape}, saved={output_path.relative_to(ROOT)}"
            )


if __name__ == "__main__":
    main()
