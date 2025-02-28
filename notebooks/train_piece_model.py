# %%
from roboflow import Roboflow
from ultralytics import YOLO
import os
from dotenv import load_dotenv

# %%
# Load environment variables
load_dotenv()

# Ensure models directory exists
os.makedirs("../models", exist_ok=True)

# Download dataset
rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
project = rf.workspace("ml-ki9ku").project("chess.com")
version = project.version(2)
dataset = version.download("yolov11", location="../dataset/chess.com")
                
# %%
# Load a pre-trained model
model = YOLO('../models/yolo11n.pt')

# Train the model
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name='chess_piece_detector'
)