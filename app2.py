import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load('rf_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the request
    features = request.json['features']

    # Convert the features to numpy array
    features_array = np.array(features).reshape(1, -1)

    # Predict the disease
    prediction = model.predict(features_array)
    # Predict the probabilities
    probabilities = model.predict_proba(features_array)

    # Define the possible diseases
    diseases = ['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc']

    # Get the disease name and probability
    disease_name = diseases[prediction[0]]
    probability = probabilities[0][prediction[0]]

    # Add a slight funny line
    funny_line = "Remember, even machines make guesses sometimes. Take it with a grain of salt, and maybe some pepper for good measure!"

    # Return the prediction and probability
    return jsonify({'prediction': disease_name, 'probability': probability, 'funny_line': funny_line})

if __name__ == '__main__':
    app.run(debug=True)
