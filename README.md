# Medical Insurance Cost Prediction Models

This repository provides deep learning regression models built with TensorFlow and Keras to predict individual medical insurance charges . Trained on the benchmark `insurance.csv` dataset, the Multi-Layer Perceptron architecture processes 11 preprocessed demographic and lifestyle features . The network maps patient profiles to continuous medical cost estimates using hidden layers optimized with Mean Absolute Error and the Adam algorithm . Three saved model formats are included: lightweight weights for instant deployment (`V1.h5`, `v2.h5`) and a complete checkpoint with optimizer states for further training (`v3.h5`) .

---

## 📊 Dataset Overview

The dataset originates from the [Machine Learning with R datasets repository](https://github.com/stedy/Machine-Learning-with-R-datasets.git). It contains personal demographic, physical, and lifestyle characteristics of primary beneficiaries to estimate their billed medical expenses.

* **Target Variable (`charges`)**: Individual medical costs billed by health insurance (continuous regression target) .
* **Input Features (`11` dimensions)**: Derived from preprocessing the 6 raw dataset features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) . Preprocessing involves standardizing continuous variables and one-hot encoding categorical variables.

---

## 🧠 Model Architecture & Specifications

All three saved model files share a consistent **Multi-Layer Perceptron (MLP)** architecture built using **TensorFlow / Keras (v3.13.2)** :

| Layer | Type | Output Units | Activation | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Input Layer** | `InputLayer` | `11` features | — | Receives preprocessed feature vector |
| **Hidden Layer 1** | `Dense` | `100` | Linear / ReLU | High-dimensional feature representation |
| **Hidden Layer 2** | `Dense` | `10` | Linear / ReLU | Dimensionality reduction & feature synthesis |
| **Output Layer** | `Dense` | `1` | Linear | Continuous regression forecast for `charges` |

### Training Configuration
* **Loss Function**: Mean Absolute Error (`MAE`) 
* **Evaluation Metric**: Mean Absolute Error (`mae`) 
* **Optimizer**: Adam ($\text{learning\_rate} = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$) 

---

## 📁 Model Files Breakdown

| File Name | Format | Contents | Best Suited For |
| :--- | :--- | :--- | :--- |
| **`V1.h5`** | HDF5 | Architecture configuration + trained weights (`top_level_model_weights`) | **Production Inference**: Clean lightweight deployment for APIs or web services without training overhead. |
| **`v2.h5`** | HDF5 | Architecture configuration + trained weights (`top_level_model_weights`) | **Lightweight Evaluation**: Direct prediction workflows and testing against baseline performance. |
| **`v3.h5`** | HDF5 | Complete training package: architecture + weights + **Adam optimizer state** (`optimizer_weights`) | **Active Development**: Resuming training, fine-tuning checkpoints, or transfer learning without losing optimizer momentum. |

---

## 🚀 Quick Start & Usage

You can load and execute inference with the models using standard TensorFlow/Keras code in Python:

```python
import numpy as np
import tensorflow as tf

# 1. Load the saved model (choose V1.h5, v2.h5, or v3.h5)
model = tf.keras.models.load_model("v3.h5")

# 2. Prepare sample preprocessed input vector (shape: [batch_size, 11])
# Features: standardized age/bmi/children + one-hot encoded sex, smoker, and region
sample_input = np.zeros((1, 11), dtype=np.float32)

# 3. Predict insurance charges
predicted_charge = model.predict(sample_input)
print(f"Predicted Medical Insurance Charge: ${predicted_charge[0][0]:.2f}")
