# %%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# %%
# Helper function to display images in notebook
def show_image(img, title='Image', figsize=(10,10)):
    plt.figure(figsize=figsize)
    # # Convert BGR to RGB for matplotlib
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load and display original image
image_path = "../dataset/screenshots/image.png"
original = cv.imread(image_path)
show_image(original, "Original Image")

# %%
# 1. Preprocessing cell
def preprocess_image(img):
    # Convert to grayscale but keep the original for later use
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Gaussian blur: (5,5) is kernel size, 0 is sigma
    # Smaller kernel = less blurring. You can try (3,3) for less blur
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    
    # Show intermediate steps
    show_image(gray, "Grayscale Image")
    show_image(blurred, "After Gaussian Blur")
    
    return blurred  # Return blurred image instead of threshold

processed = preprocess_image(original)
show_image(processed, "Processed Image")

# %%
# 2. Chessboard Detection cell
def find_chessboard_corners(img):
    # Find edges using Canny edge detection
    edges = cv.Canny(img, 50, 150)
    show_image(edges, "Edge Detection")
    
    # Find contours in the edge image
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours on a copy of the original image
    contour_img = original.copy()
    cv.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    show_image(contour_img, "All Contours")
    
    # Find the largest contour (likely to be the chessboard)
    largest_contour = max(contours, key=cv.contourArea)
    
    # Draw only the largest contour
    largest_contour_img = original.copy()
    cv.drawContours(largest_contour_img, [largest_contour], -1, (0, 255, 0), 2)
    show_image(largest_contour_img, "Largest Contour")
    
    return largest_contour

largest_contour = find_chessboard_corners(processed)

# %%
# 3. Square Detection cell
def split_chessboard(img, contour):
    # Get bounding rectangle of the contour
    x, y, w, h = cv.boundingRect(contour)
    
    # Extract the chessboard region
    chessboard = original[y:y+h, x:x+w]
    show_image(chessboard, "Extracted Chessboard")
    
    # Calculate square size
    square_width = w // 8
    square_height = h // 8
    
    # Create a visualization of the grid
    grid_img = chessboard.copy()
    for i in range(8):
        for j in range(8):
            # Draw rectangle for each square
            top_left = (j * square_width, i * square_height)
            bottom_right = ((j + 1) * square_width, (i + 1) * square_height)
            cv.rectangle(grid_img, top_left, bottom_right, (0, 255, 0), 1)
    
    show_image(grid_img, "Chessboard Grid")
    
    # Store individual squares (optional visualization)
    squares = []
    for i in range(8):
        row = []
        for j in range(8):
            square = chessboard[i*square_height:(i+1)*square_height, 
                              j*square_width:(j+1)*square_width]
            row.append(square)
        squares.append(row)
    
    # Show a few example squares (optional)
    plt.figure(figsize=(15,3))
    for i in range(5):  # Show first 5 squares of the top row
        plt.subplot(1,5,i+1)
        square_img = cv.cvtColor(squares[0][i], cv.COLOR_BGR2RGB)
        plt.imshow(square_img)
        plt.axis('off')
    plt.show()
    
    return squares

squares = split_chessboard(original, largest_contour)

# %%
# 4. Piece Detection and Classification cell

# Load your trained model
model = YOLO('../models/chess_piece_detector/weights/best.pt')

def detect_pieces_whole_board(squares):
    # 1. Reconstruct full chessboard image
    board_size = squares[0][0].shape[0] * 8
    full_board = np.zeros((board_size, board_size, 3), dtype=np.uint8)
    
    # Combine squares into full board
    for i in range(8):
        for j in range(8):
            y_start = i * squares[0][0].shape[0]
            x_start = j * squares[0][0].shape[1]
            full_board[y_start:y_start + squares[0][0].shape[0], 
                      x_start:x_start + squares[0][0].shape[1]] = squares[i][j]
    
    # 2. Run detection on full board
    predictions = model(full_board, conf=0.3)[0]
    
    # 3. Initialize results array
    results = [['empty' for _ in range(8)] for _ in range(8)]
    
    # 4. Process each detection
    for box in predictions.boxes:
        # Get box coordinates and convert to square indices
        x, y, w, h = box.xywh[0]  # Get center x, y and width, height
        square_size = full_board.shape[0] / 8
        
        # Calculate which square this detection belongs to
        col = int(x.item() / square_size)
        row = int(y.item() / square_size)
        
        if 0 <= row < 8 and 0 <= col < 8:  # Ensure within bounds
            class_name = predictions.names[int(box.cls[0])]
            results[row][col] = class_name
    
    return results

# Visualize results
def show_detections(squares, results):
    plt.figure(figsize=(20,20))
    for i in range(8):
        for j in range(8):
            plt.subplot(8,8,i*8+j+1)
            square_img = cv.cvtColor(squares[i][j], cv.COLOR_BGR2RGB)
            plt.imshow(square_img)
            plt.title(f'{results[i][j]}', pad=2)
            plt.axis('off')
    plt.tight_layout()
    plt.show()

# Use the model
piece_positions = detect_pieces_whole_board(squares)
show_detections(squares, piece_positions)

# %%
