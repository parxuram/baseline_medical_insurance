Medical Insurance Cost Prediction ModelsThis repository contains trained neural network models designed to predict individual medical insurance charges based on personal demographic and health characteristics. The models are trained on the benchmark Medical Insurance Dataset (insurance.csv).Dataset OverviewThe data originates from the Machine Learning with R datasets repository. It includes information on primary beneficiaries and predicts their billed medical expenses.Target Variable: charges (Continuous individual medical costs billed by health insurance)Input Shape: (None, 11) — Results from preprocessing the 6 raw features (age, sex, bmi, children, smoker, region) via one-hot encoding for categorical variables and feature scaling.Model Architecture & SpecificationsAll three saved model files (V1.h5, v2.h5, and v3.h5) share a consistent Multi-Layer Perceptron (MLP) architecture built using TensorFlow / Keras (v3.13.2) :LayerTypeOutput UnitsActivationPurposeInput LayerInputLayer11 features—Accepts processed demographic/health vectorsHidden Layer 1Dense100Linear / ReLUHigh-dimensional feature representationHidden Layer 2Dense10Linear / ReLUDimensionality reduction and pattern consolidationOutput LayerDense1LinearContinuous regression forecast for chargesTraining ConfigurationLoss Function: Mean Absolute Error (MAE)Evaluation Metric: Mean Absolute Error (MAE)Optimizer: Adam ($\text{learning\_rate} = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$)Model Files SummaryV1.h5 & v2.h5: Contain the saved structural definitions and model weights (top_level_model_weights) . Ideal for lightweight inference pipelines where optimizer state is unneeded.v3.h5: Full model package containing the architecture configuration, learned layer weights, and complete Adam optimizer state (optimizer_weights / momentum vectors) . Best suited if you intend to resume training or fine-tune checkpoints .Quick Start / UsageYou can load and evaluate the models using standard TensorFlow/Keras syntax in Python:Pythonimport numpy as np
import tensorflow as tf

# 1. Load the trained model
model = tf.keras.models.load_model("v3.h5")

# 2. Prepare sample input data (11 features after preprocessing/one-hot encoding)
# Example vector: [age, bmi, children, sex_male, smoker_yes, region_nw, region_se, region_sw, ...]
sample_input = np.zeros((1, 11), dtype=np.float32)

# 3. Predict insurance charges
predicted_charge = model.predict(sample_input)
print(f"Predicted Medical Insurance Charge: ${predicted_charge[0][0]:.2f}")
