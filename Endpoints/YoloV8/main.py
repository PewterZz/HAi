import json
import time
import requests
from PIL import Image, ImageDraw

start_time = time.time()
url = "https://api.ultralytics.com/v1/predict/1GCDYisYuPV1rQ100wE4"
headers = {"x-api-key": "76fc625b7251b36ad152cb916446b3bf555672d444"}
data = {"size": 640, "confidence": 0.25, "iou": 0.45}

image_path = "F:\\Projects\\HAi\\data\\images\\test\\1as.jpg"
with open(image_path, "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"image": f})

response.raise_for_status()

response_data = response.json()
end_time = time.time()

size_of_data = len(response_data) + (1024 * 500)
time_taken = end_time - start_time

bandwidth = size_of_data / time_taken  

image = Image.open(image_path)

draw = ImageDraw.Draw(image)
for item in response_data["data"]:
    box = item["box"]
    x1, x2 = box["x1"], box["x2"]
    y1, y2 = box.get("y1", 0), box.get("y2", image.height)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    print("name: ", item['name'])

print(f"Bandwidth: {bandwidth} bytes/second")
image.show()
