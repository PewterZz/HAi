from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from PIL import Image
import numpy as np
import io

model = load_model('./models/fer_emotion.h5')

app = FastAPI()

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  
    image = np.array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    return {"prediction": prediction.tolist()}

