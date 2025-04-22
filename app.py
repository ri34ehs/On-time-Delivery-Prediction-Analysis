from flask import Flask, jsonify, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model
with open('diabetics_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the saved scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the request
    features = [data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                data['SkinThickness'], data['Insulin'], data['BMI'],
                data['DiabetesPedigreeFunction'], data['Age']]

    # Convert features to a 2D array and scale them
    input_array = np.array([features])
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        return jsonify({'prediction': "Has diabetes"})
    else:
        return jsonify({'prediction': "No diabetes"})

if __name__ == "__main__":
    app.run(debug=True)
