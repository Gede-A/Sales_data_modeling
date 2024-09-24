from flask import Flask, request, jsonify
from app.model import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    prediction = predict(data['input'])
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
