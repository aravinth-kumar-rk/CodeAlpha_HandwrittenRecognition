# CodeAlpha Handwritten Character Recognition (Task 3)

An end-to-end Convolutional Neural Network (CNN) architecture built to recognize and classify handwritten characters with high accuracy. Developed as part of the CodeAlpha Machine Learning Internship.

## 🚀 Project Overview
This project processes image pixel arrays to accurately identify handwritten characters. It implements deep learning principles using a custom CNN pipeline optimized for image feature extraction and multi-class classification.

* **Architecture:** Convolutional Neural Network (CNN)
* **Framework:** TensorFlow / Keras / Python
* **Environment:** Google Colab / Local GPU

---

## 📊 Key Components & Workflow
1. **Data Preprocessing:** Normalizing pixel values, reshaping image dimensions, and one-hot encoding target labels.
2. **CNN Architecture:** 
   * Convolutional layers with ReLU activation for feature extraction.
   * Max-pooling layers for spatial dimension reduction.
   * Flatten and Dense layers for final classification.
3. **Model Training & Evaluation:** Optimized using categorical cross-entropy and evaluated on validation loss and accuracy metrics.

---

## 🛠️ Repository Structure
* `CodeAlpha_HandwrittenCharacterRecognition.ipynb` - Step-by-step interactive notebook covering data loading, model building, training history, and evaluation outputs.
* `CodeAlpha_HandwrittenCharacterRecognition.py` - Standalone production script for direct execution.

---

## 💻 Tech Stack
* **Language:** Python 3
* **Libraries:** `tensorflow`, `keras`, `numpy`, `matplotlib`, `pandas`
