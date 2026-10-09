import os
import json
import gdown
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

from input_guard import InputDomainGuard
from src.gradcam import find_last_conv_layer, generate_gradcam_heatmap, overlay_gradcam

st.set_page_config(page_title="Malaria Cell Detection", layout="wide")

MODEL_FILES = {
    "custom_cnn_weights_only.weights.h5": "18wJ49TfpXiZksOOLSrfveyWws_VtEw9y",
    "mobilenetv2_frozen_weights_only.weights.h5": "1rwTZRkq5gOqwajVOdinK6E58ALGBQ4uZ",
}
MODEL_DIR = "models"
CLASS_NAMES = ["Parasitized", "Uninfected"]


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, file_id in MODEL_FILES.items():
        output_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=True)


def build_custom_cnn(input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="custom_conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="custom_conv2")(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Custom_CNN")


def build_mobilenetv2_frozen():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name="MobileNetV2_Frozen")


def preprocess_for_model(img, model_name):
    img_resized = img.resize((128, 128))
    arr = np.array(img_resized)
    if "MobileNetV2" in model_name:
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


@st.cache_resource
def load_models():
    download_models()

    loaded_models = {}

    # ---------------------------------------------------------
    # CUSTOM CNN
    # ---------------------------------------------------------
    custom_path = os.path.join(
        MODEL_DIR,
        "custom_cnn_weights_only.weights.h5"
    )

    try:
        custom_model = build_custom_cnn()
        custom_model.load_weights(custom_path)

        loaded_models["Custom CNN"] = custom_model

    except Exception as err:
        st.warning(
            "Custom CNN could not be loaded because its saved weights "
            "do not match the current Custom CNN architecture."
        )
        st.caption(f"Custom CNN loading error: {err}")

    # ---------------------------------------------------------
    # MOBILENETV2
    # ---------------------------------------------------------
    mobile_path = os.path.join(
        MODEL_DIR,
        "mobilenetv2_frozen_weights_only.weights.h5"
    )

    try:
        mobilenet_frozen = build_mobilenetv2_frozen()
        mobilenet_frozen.load_weights(mobile_path)

        loaded_models["MobileNetV2 Frozen"] = mobilenet_frozen

    except Exception as err:
        st.error(
            "MobileNetV2 could not be loaded. "
            "The application cannot continue without a valid model."
        )
        st.exception(err)
        st.stop()

    return loaded_models

@st.cache_resource
def load_input_guard(model):
    return InputDomainGuard(model, max_reference_images=80)


with st.spinner("Loading verified models..."):
    models_dict = load_models()

# Use the MobileNetV2 model as the domain representation because it has
# ImageNet-pretrained convolutional features followed by the project's
# malaria-specific 128-unit embedding layer.
guard = load_input_guard(models_dict["MobileNetV2 Frozen"])

st.title("Malaria Cell Detection App")
st.caption("Microscopic thin blood smear classification with input-domain rejection and Grad-CAM explainability.")

if guard.detector is not None:
    st.success(
        f"Input-domain validation: ACTIVE — {guard.reference_count} genuine cell images "
        f"used as the reference domain."
    )
else:
    st.warning(
        "Input-domain validation is waiting for the malaria dataset. "
        "The classifier will NOT be run until genuine reference cell images are available."
    )

st.info(
    "Upload one individual stained blood-smear cell image. "
    "Images outside the learned microscopy-cell domain are rejected before classification. "
    "This is a research/academic demonstration, not a certified medical diagnostic device."
)

tab_inference, tab_metrics = st.tabs(["Inference & Grad-CAM", "Model Performance & Metrics"])

with tab_inference:
    uploaded_file = st.file_uploader(
        "Upload Blood Smear Cell Image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_col, cam_col = st.columns(2)
        with img_col:
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        # ---------------------------------------------------------
        # STAGE 1: DOMAIN VALIDATION
        # ---------------------------------------------------------
        accepted, gate_message, gate_info = guard.validate(pil_image)

        if not accepted:
            with cam_col:
                st.error("Image rejected")
                st.write(gate_message)

            st.subheader("Input Validation")
            st.warning(
                "The malaria classifier was NOT executed because this image "
                "did not pass the microscopy-cell domain gate."
            )
            st.caption(
                "The gate uses image structure and, when the dataset is available, "
                "a one-class detector fitted to genuine malaria-cell embeddings."
            )
            st.stop()

        st.success("Input accepted as a microscopy-cell-domain image.")

        # ---------------------------------------------------------
        # STAGE 2: SELECTED MODEL CLASSIFICATION
        # ---------------------------------------------------------
        selected_model_name = st.selectbox(
            "Select model:",
            [name for name in ("Custom CNN", "MobileNetV2 Frozen") if name in models_dict],
        )
        selected_model = models_dict[selected_model_name]
        tensor_input = preprocess_for_model(pil_image, selected_model_name)
        raw_prob = float(selected_model(tensor_input, training=False)[0][0])

        # Directory labels are Parasitized=0 and Uninfected=1.
        p_uninfected = raw_prob
        p_parasitized = 1.0 - raw_prob
        class_index = 1 if p_uninfected >= p_parasitized else 0
        pred_label = CLASS_NAMES[class_index]
        confidence = max(p_parasitized, p_uninfected)

        st.subheader("Selected Model Prediction")
        st.write(f"Model: **{selected_model_name}**")
        st.write(f"Prediction: **{pred_label}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        st.write(f"Parasitized: {p_parasitized:.2%} · Uninfected: {p_uninfected:.2%}")
        if confidence < 0.65:
            st.warning("The image passed the domain gate, but this model has low prediction confidence.")

        # ---------------------------------------------------------
        # STAGE 3: GRAD-CAM EXPLANATION
        # ---------------------------------------------------------
        try:
            target_layer_name = find_last_conv_layer(selected_model)
            heatmap = generate_gradcam_heatmap(
                selected_model,
                tensor_input,
                target_layer_name,
                class_index=class_index,
            )
            overlay = overlay_gradcam(pil_image, heatmap, alpha=0.4)
            with cam_col:
                st.image(
                    overlay,
                    caption=f"Grad-CAM overlay ({selected_model_name})",
                    use_container_width=True,
                )
            st.caption(
                "The Grad-CAM visualization provides model interpretability by highlighting "
                "image regions contributing to CNN predictions. It is intended for model "
                "interpretation and not clinical diagnosis."
            )
        except Exception as err:
            st.warning(f"Grad-CAM could not be generated: {err}")

with tab_metrics:
    st.subheader("Measured Model Evaluation")
    results_path = os.path.join("results", "model_evaluation_results.json")
    if not os.path.isfile(results_path):
        st.info("Evaluation results are not available yet. Run `python evaluate_models.py` to measure the local models.")
    else:
        try:
            with open(results_path, "r", encoding="utf-8") as result_file:
                evaluation = json.load(result_file)
            st.caption(
                f"Dataset: {evaluation['dataset']} · {evaluation['images_found']} validation images · "
                f"Class mapping: {evaluation['class_mapping']}"
            )
            for model_name, metrics in evaluation.get("models", {}).items():
                st.markdown(f"### {model_name}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2%}", border=True)
                col2.metric("Precision (Parasitized)", f"{metrics['precision']:.3f}", border=True)
                col3.metric("Sensitivity (Parasitized)", f"{metrics['recall_sensitivity']:.3f}", border=True)
                col4.metric("Specificity (Uninfected)", f"{metrics['specificity']:.3f}", border=True)
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("F1-score", f"{metrics['f1_score']:.3f}", border=True)
                col6.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", border=True)
                col7.metric("Parameters", f"{metrics['parameters']:,}", border=True)
                col8.metric("Inference ms/image", f"{metrics['inference_time_ms_per_image']:.3f}", border=True)
                st.caption(f"Input preprocessing: {metrics['preprocessing']}")
                matrix_path = metrics.get("confusion_matrix_file")
                if matrix_path and os.path.isfile(matrix_path):
                    st.image(matrix_path, caption=f"{model_name} confusion matrix", width=480)
                st.divider()
        except (OSError, ValueError, KeyError, TypeError) as err:
            st.error(f"Could not read the evaluation results: {err}")
