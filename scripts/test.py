import cv2
import torch
from torchvision import transforms
from torchvision.models import googlenet
from PIL import Image
import numpy as np


def load_model(model_path):
    model = googlenet(pretrained=False, num_classes=1000, aux_logits=False)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    return model


def process_image(frame):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(frame)
    image = transform(image).unsqueeze(0)
    return image

def predict(model, processed_frame):
    with torch.no_grad():
        output = model(processed_frame)
        _, predicted = torch.max(output, 1)
        return predicted

def camera_detection(model):
    cap = cv2.VideoCapture(0) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_image(frame)
        prediction = predict(model, processed_frame)
        cv2.putText(frame, f'Prediction: {prediction.item()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model_path = './models/cocodetect.pth'

model = load_model(model_path)

camera_detection(model)
