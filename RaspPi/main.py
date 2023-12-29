import json
import requests
import pygame
import picamera
import time
import os

# API endpoint and API key
url = "https://api.ultralytics.com/v1/predict/1GCDYisYuPV1rQ100wE4"
headers = {"x-api-key": "76fc625b7251b36ad152cb916446b3bf555672d444"}

# Camera configuration
camera = picamera.PiCamera()
camera.resolution = (640, 480)

# Set the directory where the image will be saved
image_dir = "/home/pi/Pictures"  # Change this path to your desired directory
os.makedirs(image_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Capture an image and save it in the "Pictures" directory
image_filename = "captured_image.jpg"
image_path = os.path.join(image_dir, image_filename)
camera.start_preview()

# Wait for the camera to warm up
time.sleep(2)

# Capture an image
camera.capture(image_path)
camera.stop_preview()
camera.close()

# Prediction data
data = {"size": 640, "confidence": 0.25, "iou": 0.45}

# Make API request with the captured image
with open(image_path, "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"image": f})

response.raise_for_status()

# Get the prediction result
prediction_result = response.json()
print(json.dumps(prediction_result, indent=2))

# Initialize Pygame for text-to-speech
pygame.mixer.init()
pygame.mixer.music.load("/usr/share/sounds/alsa/Front_Center.wav")  # Replace with your desired audio file

# Say the prediction out loud
for prediction in prediction_result["data"]:
    text_to_speech = f"I predict this is a {prediction['name']} with confidence {prediction['confidence']:.2f}"
    pygame.mixer.music.play()
    pygame.mixer.music.set_volume(1.0)
    time.sleep(1)  # You can adjust the delay as needed
    print(text_to_speech)
    pygame.mixer.music.stop()

# Clean up
pygame.mixer.quit()
