from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import librosa
import numpy as np
import pickle
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

# Load the emotion recognition model and related resources
emotion_model = pickle.load(open('../../models/ravdess/model.pkl', 'rb'))
scaler = pickle.load(open('../../models/ravdess/scaler.pkl', 'rb'))
encoder = pickle.load(open('../../models/ravdess/encoder.pkl', 'rb'))

# Load the image-based emotion recognition model
image_model = load_model('../../models/fer_emotion.h5')

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

@app.post("/audio/predict/")
async def predict_audio(file: UploadFile = File(...)):
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
    prediction = emotion_model.predict(features)
    new_emotion = encoder.inverse_transform(prediction)[0][0]
    
    # Return the prediction as a JSON response
    return JSONResponse(content={"prediction": new_emotion})

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  
    image = np.array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

@app.post("/image/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    processed_image = preprocess_image(image)

    prediction = image_model.predict(processed_image)

    return {"prediction": prediction.tolist()}

@app.get("/test")
async def test():
    return { "msg": "Hello Peter" }