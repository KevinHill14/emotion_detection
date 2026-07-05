# Facial Emotion Detector

A convolutional neural network (CNN) trained from scratch on the FER2013 dataset to classify facial emotions in real time via webcam. Built as a personal project to learn the fundamentals of machine learning from backpropagation math to a deployed live application.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green)

## Overview

This project trains a CNN to classify 7 emotions — **angry, disgust, fear, happy, neutral, sad, surprise** — from 48x48 grayscale facial images, then deploys the trained model in a live webcam application using OpenCV for face detection.

The dataset (FER2013) is small, imbalanced, and known to be genuinely difficult. Other models accuracy cap around 65%, and published state-of-the-art single-model results sit around 72–75%. This project's goal was to build a solid, honestly-validated model from first principles, not to chase a leaderboard number.

## Results

| Model | Test Accuracy |
|---|---|
| Single model (best individual CNN) | **61%** |
| 5-model ensemble (soft voting) | **65%** |

Per-class accuracy on the final ensemble, evaluated once on a test set never seen before:

| Emotion | Accuracy |
|---|---|
| Angry | 55.6% |
| Disgust | 62.2% |
| Fear | 49.6% |
| Happy | 83.0% |
| Neutral | 61.7% |
| Sad | 53.4% |
| Surprise | 75.8% |

For context: FER2013's "disgust" class has only ~440 training images (roughly 16x fewer than "happy"), and an untuned baseline model classified it correctly only ~11% of the time. Every class in the final model clears 45% accuracy, my personal goal.

## Architecture

- 4 convolutional blocks (32 → 64 → 128 → 256 filters), each with BatchNormalization and MaxPooling
- Dense(256) classifier head with L2 regularization and Dropout
- Softmax output over 7 classes
- Trained with class weighting to correct for FER2013's severe class imbalance, label smoothing, and a learning-rate scheduler

## Techniques used

- **Class weighting** — tuned per-class to address FER2013's imbalance without collapsing majority-class accuracy
- **Data augmentation** — random flip, rotation, and zoom to reduce overfitting
- **Batch normalization** — stabilizes training across deeper layers
- **L2 regularization + Dropout** — combined to control overfitting as model depth increased
- **Learning rate scheduling** (`ReduceLROnPlateau`) and **early stopping**
- **Model ensembling** — 5 independently trained models combined via soft voting (averaged softmax probabilities), each with slightly different regularization/augmentation settings to encourage complementary strengths and weaknesses

## Known limitations

- **Fear vs. surprise confusion**: these two emotions share very similar facial features (wide eyes, raised brows), and the model — like most FER2013-trained models — frequently confuses them, especially in live webcam conditions.
- **Live accuracy is lower than benchmark accuracy**: webcam lighting, camera quality, and framing differ meaningfully from FER2013's static training photos, which affects real-world performance versus the reported test numbers above.
- **Disgust remains the hardest class to measure precisely**, simply due to how few examples exist in the dataset (statistically noisy even after weighting fixes).

## Project structure

```
emotion_detector/
├── data/                  # FER2013 dataset (not tracked in git)
├── models/                # Trained model files (.keras)
├── src/
│   ├── train.py           # Data loading, model definition, training
│   ├── webcam_detect.py   # Live webcam inference (single model)
│   └── webcam_detect_ensemble.py  # Live webcam inference (5-model ensemble)
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/KevinHill14/emotion_detector.git
cd emotion_detector
```

**2. Create and activate a virtual environment (Python 3.11)**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Get FER2013 from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013), extract it, and place the `train/` and `test/` folders inside `data/`.

**5. Train**
```bash
python src/train.py
```

**6. Run live detection**
```bash
python src/webcam_detect.py
```

## What I learned

This project started with hand-deriving backpropagation on a two-neuron network before writing a single line of TensorFlow. Key concepts explored along the way:

- The math behind forward passes, gradient descent, and backpropagation (chain rule, applied layer by layer)
- Why convolution and pooling exist, and what a "feature map" actually represents
- Class imbalance handling and the real tradeoffs of class weighting
- The difference between validation and test data — and the consequences of not respecting that separation during iterative tuning
- Regularization (Dropout, L2, BatchNorm) and their distinct roles in controlling overfitting
- Model ensembling and why averaging independently trained models improves robustness
- Deploying a trained model in a real-time OpenCV application, including face detection tuning (Haar cascades) separate from the classifier itself

## Acknowledgments

Trained on the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), originally introduced for the ICML 2013 Challenges in Representation Learning.
