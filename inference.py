import requests
import json
import cv2
import time
import mlflow

img_path = "/Users/aashishmukund/Downloads/glasses-min.png"  # Replace with your image
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640)).astype("float32") / 255.0
img = img.transpose(2, 0, 1).reshape(1, 3, 640, 640)

# Convert to JSON format
data = json.dumps({"instances": img.tolist()})

# Send request
response = requests.post("http://127.0.0.1:5000/invocations", data=data, headers={"Content-Type": "application/json"})

# Print response
print(response.json())

#logging the metrics
with mlflow.start_run():
    start_time = time.time()
    
    response = requests.post("http://127.0.0.1:5000/invocations", json=data)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Log inference time in MLflow
    mlflow.log_metric("Inference Time", inference_time)
    
    print("Inference Time:", inference_time, "seconds")
