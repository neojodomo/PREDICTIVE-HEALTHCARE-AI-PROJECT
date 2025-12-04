
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = '../model_dev/heart_disease_model.pkl'

@app.route('/')
def home():
    return "Heart Disease Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Model not found. Run train_model.py first.'}), 500
            
        model = joblib.load(MODEL_PATH)
        data = request.json
        # Convert JSON to array (Simplified for example)
        features = np.array([data['features']]) 
        prediction = model.predict(features)
        
        return jsonify({'risk_prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
