from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
model = pickle.load(open('model.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))

encoder = pickle.load(open('encoder.pkl', 'rb'))

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def extract_features(data, path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) 

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) 

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) 

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) 
    
    return result

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    res1 = extract_features(data, path)
    result = np.array(res1)
    
    noise_data = noise(data)
    res2 = extract_features(noise_data, path)
    result = np.vstack((result, res2)) 
  
    return result

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Save the uploaded audio file
    with open(file.filename, "wb") as audio_file:
        audio_file.write(file.file.read())

    # Process the audio file
    audio_data, sample_rate = librosa.load(file.filename, sr=None)
    features = get_features(file.filename)
    
    # Transform features using the loaded scaler
    features = scaler.transform(features)
    features = np.expand_dims(features, axis=2)

    # Make predictions using the machine learning model
    prediction = model.predict(features)
    new_emotion = encoder.inverse_transform(prediction)[0][0]
    
    # Return the prediction as a JSON response
    return JSONResponse(content={"prediction": new_emotion})
