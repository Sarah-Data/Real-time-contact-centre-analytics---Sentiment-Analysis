from flask import Flask, render_template, request
import pickle
import tensorflow
import librosa
import numpy as np
import pandas as pd
import torch
import keras as keras
from sklearn.preprocessing import LabelEncoder
from IPython.display import Audio
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load the ML model and label encoder
with open('NNtrainedmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle audio recording and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the recorded audio file from the form
    audio_file = request.files['audio']

    # Save the audio file to a temporary location
    audio_path = 'temp.wav'
    audio_file.save(audio_path)

    # Load the audio file and extract features
    audio, sr = librosa.load(audio_path, sr=None)

    # Define the functions for feature extraction
    def get_zcr(audio, frame_length=2048, hop_length=512):
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(zcr)

    def get_rmse(audio, frame_length=2048, hop_length=512):
        rmse = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(rmse)

    def get_mfcc(audio, sr, frame_length=2048, hop_length=512, flatten=True):
        mfcc_feature = librosa.feature.mfcc(y=audio, sr=sr)
        return np.ravel(mfcc_feature.T) if flatten else mfcc_feature.T

    # Extract features from the audio
    zcr_feature = get_zcr(audio)
    rmse_feature = get_rmse(audio)
    mfcc_feature = get_mfcc(audio, sr)

    # Reduce the shape of the features to the desired length
    zcr_feature = zcr_feature[:284]
    rmse_feature = rmse_feature[:284]
    mfcc_feature = mfcc_feature[:5680]

    # Create a DataFrame for the features
    dftest = pd.DataFrame({'ZCR': [zcr_feature], 'RMSE': [rmse_feature], 'MFCC': [mfcc_feature]})

    # Convert NumPy arrays to tensors
    
    dftest['ZCR'] = dftest['ZCR'].apply(lambda x: torch.tensor(x))
    dftest['RMSE'] = dftest['RMSE'].apply(lambda x: torch.tensor(x))
    dftest['MFCC'] = dftest['MFCC'].apply(lambda x: torch.tensor(x))

    # Expand tensors into individual columns
    dftest = pd.concat([dftest.drop(['ZCR', 'RMSE', 'MFCC'], axis=1),
                dftest['ZCR'].apply(pd.Series).add_prefix('ZCR_'),
                dftest['RMSE'].apply(pd.Series).add_prefix('RMSE_'),
                dftest['MFCC'].apply(pd.Series).add_prefix('MFCC_')], axis=1)

    # Fill NaN with 0
    dftest = dftest.fillna(0)

    # Make predictions using the model
    predictions = model.predict(dftest.values)

    # Convert predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=-1)  
    #predicted_labels = label_encoder.inverse_transform(predictions)

    # Render the result page with the predicted label
    return render_template('result_.html', label=predicted_labels)

# Run the Flask application
if __name__ == '__main__':
    app.run(port=8000, debug=True)
