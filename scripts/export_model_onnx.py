import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

def export_model():
    os.makedirs("models", exist_ok=True)
    
    try:
        accel_model = joblib.load("models/accel_gb.joblib")
        gyro_model = joblib.load("models/gyro_gb.joblib")
    except Exception as e:
        print(f"Could not load models: {e}")
        return

    # Define input shape. Assuming n_features
    try:
        n_features = accel_model.n_features_in_
    except AttributeError:
        # Fallback if scikit-learn version differences
        n_features = 50 # Example window size/feature count
        
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    # Convert Accel Model
    onx_accel = convert_sklearn(accel_model, initial_types=initial_type)
    with open("models/sumaro_accel_correction.onnx", "wb") as f:
        f.write(onx_accel.SerializeToString())

    # Convert Gyro Model
    onx_gyro = convert_sklearn(gyro_model, initial_types=initial_type)
    with open("models/sumaro_gyro_correction.onnx", "wb") as f:
        f.write(onx_gyro.SerializeToString())

    print("Successfully exported models to ONNX.")
    print(f"Expected input shape: [batch_size, {n_features}]")

if __name__ == "__main__":
    export_model()
