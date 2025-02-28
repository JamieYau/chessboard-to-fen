# %%
import os

from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

# %%
# Load environment variables
load_dotenv()

# Ensure models directory exists
os.makedirs("../models", exist_ok=True)

# Download dataset
rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
project = rf.workspace("2d-chess-glasses").project("chessboard-corners-v3")
version = project.version(3)
dataset = version.download("yolov11", location="../dataset/chessboards")
                
# %%
# Load a pre-trained model
model = YOLO('../models/yolo11n.pt')

# Train the model
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name='chessboard_corners'
)
