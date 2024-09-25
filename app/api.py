from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model (Ensure path is correct relative to where the script is run)
model_pipeline = joblib.load('./notebooks/model_24-09-2024-01-59-46.pkl')

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Machine Learning API!"})

@app.route('/predict', methods=['POST'])
def predict():
    # Parse incoming JSON data
    try:
        data = request.get_json()  
        input_data = data['input']  
    except (KeyError, TypeError):
        return jsonify({"error": "Invalid input format. Please provide a valid 'input' key in the JSON."}), 400

    input_data = np.array(input_data).reshape(1, -1)

    try:
        prediction = model_pipeline.predict(input_data)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
