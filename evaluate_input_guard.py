"""Measure input guard rejection on validation cells and available OOD examples.

Place labeled non-cell images under data/ood/<category>/ before running to
measure false acceptance. No synthetic OOD examples are generated.
"""

import json
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image

from evaluate_models import BATCH_SIZE, CLASS_NAMES, DATA_DIR, IMG_SIZE, MODEL_DIR, RESULT_DIR
from input_guard import InputDomainGuard


ROOT = Path(__file__).resolve().parent
OOD_DIR = ROOT / "data" / "ood"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
IN_DOMAIN_SAMPLE_PER_CLASS = 100


def main():
    weights = MODEL_DIR / "mobilenetv2_frozen_weights_only.weights.h5"
    if not weights.is_file():
        raise FileNotFoundError(f"Input guard requires the frozen MobileNetV2 weights: {weights}")

    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(128, 128, 3)
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(base.input, output, name="MobileNetV2_Frozen")
    model.load_weights(weights)
    guard = InputDomainGuard(model, max_reference_images=80)
    if guard.detector is None:
        raise RuntimeError("The input guard could not fit its detector to genuine cell references.")

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2,
    ).flow_from_directory(
        str(DATA_DIR), classes=CLASS_NAMES, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="binary", subset="validation", shuffle=False,
    )

    # Exclude reference images from the in-domain score so reference examples
    # cannot make the measured false-rejection rate artificially optimistic.
    reference_paths = {Path(path).resolve() for path in guard._collect_reference_paths()}
    by_class = {name: {"evaluated": 0, "rejected": 0} for name in CLASS_NAMES}
    embedding_batches = []
    selected = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_paths = [
            path for path, label in zip(generator.filepaths, generator.classes)
            if int(label) == class_index and Path(path).resolve() not in reference_paths
        ]
        indices = np.linspace(
            0, len(class_paths) - 1,
            min(IN_DOMAIN_SAMPLE_PER_CLASS, len(class_paths)), dtype=int,
        )
        selected.extend((class_paths[index], class_index) for index in indices)

    for path, label in selected:
        with Image.open(path) as source:
            image = source.convert("RGB")
            structure_ok, _, _ = guard._cheap_structure_check(image)
            embedding_input = guard._preprocess(image) if structure_ok else None
        outcome = by_class[CLASS_NAMES[int(label)]]
        outcome["evaluated"] += 1
        if embedding_input is None:
            outcome["rejected"] += 1
        else:
            embedding_batches.append((outcome, embedding_input))

    # Batch the exact embedding stage used by validate() for practical runtime.
    for start in range(0, len(embedding_batches), BATCH_SIZE):
        chunk = embedding_batches[start:start + BATCH_SIZE]
        batch = tf.convert_to_tensor([item[1] for item in chunk], dtype=tf.float32)
        embeddings = guard.embedding_model(batch, training=False).numpy().astype("float32")
        embeddings /= (tf.norm(embeddings, axis=1, keepdims=True).numpy() + 1e-8)
        predictions = guard.detector.predict(embeddings)
        for (outcome, _), prediction in zip(chunk, predictions):
            outcome["rejected"] += int(prediction != 1)

    for outcome in by_class.values():
        outcome["false_rejection_rate"] = (
            outcome["rejected"] / outcome["evaluated"] if outcome["evaluated"] else None
        )
    total_evaluated = sum(row["evaluated"] for row in by_class.values())
    total_rejected = sum(row["rejected"] for row in by_class.values())
    in_domain_frr = total_rejected / total_evaluated if total_evaluated else None

    ood_paths = sorted(
        path for path in OOD_DIR.rglob("*")
        if OOD_DIR.is_dir() and path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ) if OOD_DIR.is_dir() else []
    ood_rejected = 0
    for path in ood_paths:
        with Image.open(path) as source:
            accepted, _, _ = guard.validate(source.convert("RGB"))
        ood_rejected += int(not accepted)

    result = {
        "method": "App InputDomainGuard; frozen MobileNetV2 embedding detector",
        "in_domain_source": "Evenly spaced sample from 20% ImageDataGenerator validation subset",
        "in_domain_sample_per_class_requested": IN_DOMAIN_SAMPLE_PER_CLASS,
        "reference_images_excluded_from_evaluation": len(reference_paths),
        "in_domain_by_class": by_class,
        "in_domain_false_rejection_rate": in_domain_frr,
        "ood_source": str(OOD_DIR.relative_to(ROOT)).replace("\\", "/"),
        "ood_images_found": len(ood_paths),
        "ood_false_acceptances": len(ood_paths) - ood_rejected,
        "ood_false_acceptance_rate": (len(ood_paths) - ood_rejected) / len(ood_paths) if ood_paths else None,
        "ood_rate_status": "measured" if ood_paths else "unavailable: no OOD images were provided",
        "medical_validation": False,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "input_guard_evaluation_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
