from ultralytics import YOLO
import numpy as np

model = YOLO('../models/chess_piece_detector/weights/best.pt')

# ... existing code ...

def detect_pieces(board):
    # Run YOLO detection on full board
    predictions = model(board, conf=0.3)[0]
    
    # Get board dimensions
    h, w = board.shape[:2]
    square_size = h // 8
    
    # Initialize results grid with NumPy
    results = np.full((8, 8), 'empty', dtype=object)  # or dtype=str
    
    # Process detections
    for box in predictions.boxes:
        x, y, w, h = box.xywh[0]
        col = int(x.item() / square_size)
        row = int(y.item() / square_size)
        
        if 0 <= row < 8 and 0 <= col < 8:
            class_name = predictions.names[int(box.cls[0])]
            results[row, col] = class_name
    
    return results
