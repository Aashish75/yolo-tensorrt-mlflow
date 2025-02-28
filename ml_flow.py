import mlflow
import mlflow.onnx
import onnx

# Set MLflow to log locally
mlflow.set_tracking_uri("file:./mlruns")  
mlflow.set_experiment("YOLO-TensorRT")

# Define the ONNX model path
onnx_model_path = "/Users/aashishmukund/Downloads/yolov8s.onnx"

with mlflow.start_run():
    # Load ONNX model
    model = onnx.load(onnx_model_path)
    
    # Log the ONNX model inside MLflow
    mlflow.onnx.log_model(model, "yolov8_onnx")
    
    print("✅ Model logged successfully")

