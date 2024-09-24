import joblib
import numpy as np

# Load the model once when the app starts
model_pipeline = joblib.load('../model_24-09-2024-01-59-46.pkl')

def predict(input_data):
    # Ensure input data is a numpy array and reshape if needed
    input_data = np.array(input_data).reshape(1, -1)
    return model_pipeline.predict(input_data)[0]
