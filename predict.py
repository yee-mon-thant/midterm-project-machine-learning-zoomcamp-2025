"""Flask API for Diabetes Prediction"""

import pickle
from flask import Flask, request, jsonify

# Load the trained model
input_file = 'model_C=1.0_diabetes.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('diabetes')

# Threshold for classification (from precision ≈ recall analysis)
THRESHOLD = 0.38  

@app.route('/predict', methods=['POST'])
def predict_api():
    # Get JSON request
    patient = request.get_json()

    # Transform features
    X = dv.transform([patient])

    # Predict probability
    y_pred = model.predict_proba(X)[0, 1]

    # Convert probability to class label using threshold
    diabetes = y_pred >= THRESHOLD

    result = {
        'diabetes_probability': float(y_pred),
        'diabetes': bool(diabetes)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
