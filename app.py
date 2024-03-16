import uvicorn
from fastapi import FastAPI
from MeternalHealthNote import MeternalHealthNote
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

# Load the CatBoost model
with open('catboost.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)

# Define a dictionary for prediction labels
prediction_labels = {0: "high risk", 1: "low risk", 2: "mid risk"}

# Function to perform feature engineering
def perform_feature_engineering(features):
    # Append additional features based on your feature engineering logic
    features_engineered = list(features)
    features_engineered.append(features_engineered[1] - features_engineered[2])
    features_engineered.append(features_engineered[1] - features_engineered[2] + features_engineered[4])
    
    return features_engineered

# Function to make predictions
def predict_species(classifier, features):
    try:
        # Perform feature engineering
        features_engineered = perform_feature_engineering(features)

        # Convert features to numpy array
        features_array = np.array([features_engineered])

        # Make prediction using the loaded classifier
        prediction = classifier.predict(features_array)[0][0]

        # Map prediction to labels
        prediction_label = prediction_labels[prediction]

        return prediction_label
    except Exception as e:
        return str(e)

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict(data: MeternalHealthNote):
    try:
        # Extract data from the request
        Age = data.Age
        SystolicBP = data.SystolicBP
        DiastolicBP = data.DiastolicBP
        BS = data.BS
        BodyTemp = data.BodyTemp
        HeartRate = data.HeartRate

        # Prepare the features for prediction
        features = [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]

        # Make prediction
        prediction = predict_species(classifier, features)
        prediction = prediction.replace("'", "")

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

