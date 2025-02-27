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
image_path = "../dataset/screenshots/full_screenshot.png"
original = cv.imread(image_path)
show_image(original, "Original Image")

# %%
# 1. Preprocessing cell
def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    
    # Show intermediate steps
    show_image(gray, "Grayscale Image")
    show_image(blurred, "After Gaussian Blur")
    
    return blurred

processed = preprocess_image(original)
show_image(processed, "Processed Image")

# %%
# 2. Chessboard Detection cell
def find_chessboard(img):
    processed = preprocess_image(img)
    # Find edges using Canny edge detection
    edges = cv.Canny(processed, 50, 150)
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
    
    # Get bounding rectangle
    x, y, w, h = cv.boundingRect(largest_contour)
    
    # Extract the chessboard region
    chessboard = img[y:y+h, x:x+w]
    
    return chessboard

largest_contour = find_chessboard(original)


# %%
# 3. Piece Detection and Classification cell

# Load your trained model
model = YOLO('../models/chess_piece_detector/weights/best.pt')

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

def process_screenshot(image_path):
    # 1. Load full screenshot
    screenshot = cv.imread(image_path)
    show_image(screenshot, "Original Screenshot")
    
    # 2. Find and extract chessboard
    chessboard = find_chessboard(screenshot)
    show_image(chessboard, "Extracted Chessboard")
    
    # 3. Detect pieces using YOLO
    piece_positions = detect_pieces(chessboard)
    
    return piece_positions, chessboard

# Use the pipeline
positions, board = process_screenshot(image_path)

# Visualize results
def show_detections(board, results):
    h, w = board.shape[:2]
    square_size = h // 8
    
    plt.figure(figsize=(20,20))
    for i in range(8):
        for j in range(8):
            # Extract and show each square
            square = board[i*square_size:(i+1)*square_size, 
                         j*square_size:(j+1)*square_size]
            plt.subplot(8,8,i*8+j+1)
            plt.imshow(cv.cvtColor(square, cv.COLOR_BGR2RGB))
            plt.title(f'{results[i][j]}', pad=2)
            plt.axis('off')
    plt.tight_layout()
    plt.show()

show_detections(board, positions)

# %%
# 4. Create FEN from positions

def create_fen(piece_array):
    # Map piece names to FEN characters
    piece_to_fen = {
        'white-pawn': 'P',
        'white-knight': 'N',
        'white-bishop': 'B',
        'white-rook': 'R',
        'white-queen': 'Q',
        'white-king': 'K',
        'black-pawn': 'p',
        'black-knight': 'n',
        'black-bishop': 'b',
        'black-rook': 'r',
        'black-queen': 'q',
        'black-king': 'k',
        'empty': 'empty'
    }
    
    # Initialize FEN string
    fen = []
    
    # Process each row
    for row in piece_array:
        empty_squares = 0
        row_fen = []
        
        # Process each square in the row
        for piece in row:
            if piece == 'empty':
                empty_squares += 1
            else:
                # If we had empty squares before this piece, add the count
                if empty_squares > 0:
                    row_fen.append(str(empty_squares))
                    empty_squares = 0
                
                # Convert piece name to FEN character using mapping
                fen_char = piece_to_fen[piece]
                row_fen.append(fen_char)
        
        # Don't forget remaining empty squares at end of row
        if empty_squares > 0:
            row_fen.append(str(empty_squares))
        
        # Add the row to FEN
        fen.append(''.join(row_fen))
    
    # Join rows with '/' and add default values for other FEN fields
    # Assuming it's white to move, all castling available, no en passant
    full_fen = '/'.join(fen) + ' w KQkq - 0 1'
    return full_fen

def create_analysis_urls(fen):
    # URL encode the FEN string
    from urllib.parse import quote
    encoded_fen = quote(fen)
    
    # Create URLs
    lichess_url = f"https://lichess.org/analysis/{encoded_fen}"
    chesscom_url = f"https://chess.com/analysis?fen={encoded_fen}"
    
    return lichess_url, chesscom_url


    
# Create FEN and URLs
fen = create_fen(positions)
lichess_url, chesscom_url = create_analysis_urls(fen)

print(f"FEN: {fen}")
print(f"Lichess Analysis: {lichess_url}")
print(f"Chess.com Analysis: {chesscom_url}")


# %%
