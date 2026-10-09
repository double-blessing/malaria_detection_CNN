"""Grad-CAM utilities for the project's binary Keras classifiers."""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the model path to its final Conv2D layer, including nested models."""

    def search(container, prefix=""):
        for layer in reversed(container.layers):
            layer_path = f"{prefix}/{layer.name}" if prefix else layer.name
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer_path
            if isinstance(layer, tf.keras.Model):
                found = search(layer, layer_path)
                if found:
                    return found
        return None

    target = search(model)
    if target is None:
        raise ValueError("No Conv2D layer was found in the model.")
    return target


def _resolve_layer(model: tf.keras.Model, layer_path: str):
    current = model
    for part in layer_path.split("/"):
        current = current.get_layer(part)
    return current


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor,
    target_layer_name: str,
    class_index: int,
) -> np.ndarray:
    """Generate a normalized heatmap for a binary model class.

    ``image_tensor`` must already use that model's inference preprocessing.
    For this project's sigmoid output, class 0 is Parasitized (1 - sigmoid)
    and class 1 is Uninfected (sigmoid).
    """
    if class_index not in (0, 1):
        raise ValueError("class_index must be 0 (Parasitized) or 1 (Uninfected).")

    inputs = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
    if inputs.shape.rank == 3:
        inputs = tf.expand_dims(inputs, axis=0)
    if inputs.shape.rank != 4:
        raise ValueError("image_tensor must have shape (H, W, C) or (1, H, W, C).")

    target_layer = _resolve_layer(model, target_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs, training=False)
        probability_uninfected = predictions[:, 0]
        class_score = (
            1.0 - probability_uninfected
            if class_index == 0
            else probability_uninfected
        )

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise RuntimeError(f"No gradients reached target layer {target_layer_name!r}.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmaps = tf.reduce_sum(channel_weights * conv_outputs, axis=-1)
    heatmaps = tf.nn.relu(heatmaps)
    heatmaps /= tf.reduce_max(heatmaps, axis=(1, 2), keepdims=True) + 1e-8
    return heatmaps[0].numpy()


def overlay_gradcam(
    original_image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a heatmap on the original image and return an RGB PIL image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    original = np.asarray(Image.fromarray(np.asarray(original_image)).convert("RGB"))
    resized = cv2.resize(
        np.asarray(heatmap, dtype=np.float32),
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    colored_bgr = cv2.applyColorMap(
        np.uint8(np.clip(resized, 0.0, 1.0) * 255),
        cv2.COLORMAP_JET,
    )
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 1.0 - alpha, colored_rgb, alpha, 0)
    return Image.fromarray(overlay, mode="RGB")
