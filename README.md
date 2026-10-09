# Malaria Image Classification using CNN

A **Convolutional Neural Network (CNN)** based machine learning model to detect **malaria-infected blood cells** from microscopic images.
Developed for the **InnovateMU Hackathon 2025**, this project demonstrates how deep learning can assist in automated medical image diagnosis.

---

## Features at a Glance
* **Binary classification**: Parasitized vs Uninfected
* **Custom CNN** and **MobileNetV2** (frozen & fine-tuned)
* **Validation accuracy**: ~94%
* **Interactive Streamlit App**:
  * Upload images for real-time predictions
  * Display **model confidence rate**
  * Show **Grad-CAM overlays** for interpretability
* **Grad-CAM Notebook** for visualization which regions of an image influence predictions
* End-to-end pipeline: data cleaning -> training -> evaluation -> deployment

---

## Project Overview

Malaria affects millions globally, and traditional diagnosis using manual microscopy is **time-consuming and prone to human error**. This project leverages **image classification** to automate malaria detection, providing **fast, consistent, and interpretable predictions**.

---

## Project Structure

```
malaria-image-classification/
│
├── notebooks/
│   ├── 01_baseline_and_frozen_models.ipynb   # Custom CNN + MobileNetV2 frozen training
│   ├── 02_finetune_mobilenetv2.ipynb         # Fine-tuning MobileNetV2
│   ├── 03_model_evaluation.ipynb             # Model comparison and metrics
│   └── 04_grad_cam_analysis.ipynb            # Grad-CAM visualization for interpretability
│
├── results/
│   ├── all_confusion_matrices.png
│   └── app_images/
│       ├── uninfected_example.png
│       └── parasitized_example.png
│
├── data/
│   ├── raw/                                  # Raw Kaggle dataset (not tracked in git)
│   └── clean_cell_images/                     # Preprocessed images for training
│
├── models/                                   # Place pretrained weights here
├── app.py                                    # Streamlit inference app
├── requirements.txt
└── README.md
```

## Dataset

Source:
[Cell Images for Detecting Malaria - Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

- ~27,000 labeled images
- Two classes: **Parasitized**, **Uninfected**
- Images resized to **128x128**
- Cleaned dataset removes duplicate nested folders

---

## Model Details

### 1. Custom CNN (Baseline)
  * Conv2D -> MaxPooling -> Dropout -> Dense -> Sigmoid
  * Loss: Binary Cross-Entropy
  * Optimizer: Adam
  * Batch size: 32, Epochs: 10-20

### 2. MobileNetV2 (Frozen Feature Extractor)
  * Pretrained on ImageNet, **base layers frozen**
  * Custom head: GlobalAveragePooling -> Dense -> Dropout -> Dense (sigmoid)
  * Loss: Binary Cross-Entropy, Optimizer: Adam
  * Batch size: 32, Epochs: 10-15

### 3. MobileNetV2 (Fine-Tuned)
  * Top convolutional blocks **unfrozen** for domain adaptation
  * Custom head: GlobalAveragePooling -> Dense -> Dropout -> Dense (sigmoid)
  * Lower learning rate for stability
  * Batch size: 32, Epochs: 10-15

---

## Grad-CAM Notebook
  * Notebook: `04_grad_cam_analysis.ipynb`
  * Generate **visual explanations** for model predictions
  * Highlights **image regions influencing the model**
  * Supports **single-image or batch visualization**
  * Produces **heatmap overlays** for intuitive understanding
> This helps interpret why the CNN predicts a cell as parasitized or uninfected, increasing trust in the model's decision-making.

---

## Quick Start
1. Clone repo and install dependencies:
```bash
git clone <https://github.com/MaryvilleUniversity-AI/malaria-image-classification.git>
cd malaria-image-classification
pip install -r requirements.txt
```
2. Run the Streamlit app:
> **Linux / GitHub Codespaces / WSL users**: you need OpenGL for image rendering
  ```bash
  sudo apt-get update
  sudo apt-get install -y libgl1
  python3 -m streamlit run app.py
  ```
> **Windows / Mac users**:
  ```bash
  streamlit run app.py
  ```
3. Use the app:
  * Upload a blood cell image
  * View **prediction**, **confidence rate**, and **Grad-CAM overlay**

## Example Outputs

![Uninfected Cell Prediction](results/app_images/uninfected_example.png)
![Parasitized Cell Prediction](results/app_images/parasitized_example.png)

---
## Model Comparison

| Model                    | Strategy                              | Accuracy | Precision | Recall | F1    |
|--------------------------|--------------------------------------|----------|-----------|--------|-------|
| Custom CNN               | Trained from scratch                  | ~94%     | ~0.94     | ~0.94  | ~0.94 |
| MobileNetV2 (Frozen)     | Transfer Learning (Frozen Layers)     | ~91%     | ~0.91     | ~0.91  | ~0.91 |
| MobileNetV2 (Fine-Tuned) | Transfer Learning + Unfrozen Layers   | 94%   | 0.94     | 0.94  | 0.94 |

## Confusion Matrices
![Confusion Matrices](results/all_confusion_matrices.png)

### Key Observations
- The **custom CNN and fine-tuned MobileNetV2 achieved similar validation accuracy (~94%)**, indicating both models performed strongly on this dataset.
- The frozen MobileNetV2 model performed slightly worse, showing that pretrained ImageNet features alone were not fully optimal for microscopic cell images.
- Fine-tuning MobileNetV2 significantly improved performance, demonstrating the importance of adapting pretrained models to domain-specific data.

---

### Conclusion
Both the custom CNN and fine-tuned MobileNetV2 achieved high classification performance, with fine-tuning allowing the pretrained model to match the custom architecture. This suggests that while domain-specific feature learning is effective, pretrained models can achieve comparable results when properly adapted to the target domain.

---

### Performance

- Validation Accuracy: ~**94%**
- Evaluation includes:
  - Confusion Matrix
  - Precision, Recall, F1-score (classification report)

---

## Training the Model (Optional)

If you want to retrain the model from scratch:

1. Set up Kaggle API:
   - Upload `kaggle.json` to your working directory
   - Set permissions:
   ```bash
   chmod 600 kaggle.json
   ```
2. Run the notebook:

```bash
jupyter notebook notebooks/MalariaImageClassification.ipynb
```

The notebook:

- Downloads the dataset from Kaggle
- Cleans the folder structure
- Trains the CNN
- Saves the model to `models/malaria_model.keras`

> Note: Training was performed on Google Colab using an NVIDIA A100 GPU for faster experimentation.

---

## Limitations

- Trained on a single dataset (may not generalize to all microscopes)
- Binary classification only (no parasite species classification)
- No clinical validation (research/demo purpose only)

---

## Future Work

- Train on **multiple datasets for better generalization**
- Deploy as a **cloud-hosted web app**

---

## Acknowledgements

- Original project developed for the **InnovateMU 2025 Hackathon** by Daniel Lai, Ewan Poirier, Srivathsav Arumugam, and Ruth Ayele.
- This repository is a **fork maintained by the MaryvilleUniversity-AI GitHub organization**, furthering the project for research and educational purposes by an apprentice contributor.
- Tools: TensorFlow, Keras, Streamlit, Google Colab
- Dataset: Kaggle - Cell Images for Detecting Malaria

---

## Disclaimer

This project is for **research and educational purposes only** and is not intended for clinical use or medical diagnosis.
