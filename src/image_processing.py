import cv2 as cv
from ultralytics import YOLO

board_model = YOLO('../models/chessboard_corners/weights/best.pt')

def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    return blurred

def find_chessboard(img):
    # Detect board in the image
    results = board_model(img)[0]
    
    if len(results.boxes) > 0:
        # Get the box with highest confidence
        box = results.boxes[0]
        x1, y1, x2, y2 = box.xyxy[0]  # Get corner coordinates
        
        # Extract board region
        chessboard = img[int(y1):int(y2), int(x1):int(x2)]
        return chessboard
    else:
        raise Exception("No chessboard detected in image")