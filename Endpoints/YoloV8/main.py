import json
import requests

url = "https://api.ultralytics.com/v1/predict/1GCDYisYuPV1rQ100wE4"
headers = {"x-api-key": "76fc625b7251b36ad152cb916446b3bf555672d444"}
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
with open("F:\Projects\HAi\data\images\\test\\seratus.jpg", "rb") as f:
	response = requests.post(url, headers=headers, data=data, files={"image": f})

response.raise_for_status()

print(json.dumps(response.json(), indent=2))