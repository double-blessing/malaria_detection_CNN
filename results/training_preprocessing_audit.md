# Training preprocessing audit

## Training preprocessing

- **Custom CNN** (`notebooks/01_baseline_model.ipynb`, cells 8 and 11): images resized to 128 x 128 RGB. Training and validation generators both apply `rescale=1./255`. Training batches additionally use rotation (20 degrees), horizontal/vertical shifts (0.1), shear (0.2), zoom (0.1), horizontal flips, and nearest-fill. Validation uses no augmentation.
- **MobileNetV2 Frozen** (`notebooks/02_frozen_mobilenetv2.ipynb`, cells 1, 4-6): images resized to 128 x 128 RGB. Training and validation use `tensorflow.keras.applications.mobilenet_v2.preprocess_input` (raw RGB values mapped to the MobileNetV2 input range). Training batches additionally use rotation, shifts, shear, zoom, and horizontal flips; validation has no augmentation.
- **MobileNetV2 Fine-tuned** (`notebooks/03_finetune_mobilenetv2.ipynb`, cells 1, 5-6, 14): the fine-tuning generators use `preprocess_input` and a 20% validation split. No data augmentation is configured in this fine-tuning notebook. No fine-tuned model weights are present in `models/`, so it is excluded from evaluation.

## App inference preprocessing

`app.py::preprocess_for_model` resizes uploaded images to 128 x 128 RGB. The Custom CNN receives pixel values divided by 255; MobileNetV2 receives `mobilenet_v2.preprocess_input` applied to the 0-255 RGB array. This matches the respective training and validation preprocessing.

## Mismatches and dataset handling

The previous `evaluate_models.py` used `rescale=1./255` for both models. That was correct for the Custom CNN and incorrect for both MobileNetV2 variants. The evaluator now creates model-specific generators and explicitly selects `Parasitized` then `Uninfected`, preserving the mapping `{'Parasitized': 0, 'Uninfected': 1}`. The inspected dataset has 13,779 image files in each class folder and no nested `cell_images/cell_images/` directory. Any other directory at the dataset root is excluded by the explicit class list.

The evaluator measures the notebook-compatible 20% validation subset, not a separate held-out test set. Treat the resulting metrics as validation measurements.
