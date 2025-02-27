import cv2 as cv
from image_processing import find_chessboard
from piece_detection import detect_pieces

def process_screenshot(image_path):
    screenshot = cv.imread(image_path)
    chessboard = find_chessboard(screenshot)
    piece_positions = detect_pieces(chessboard)
    return piece_positions

positions = process_screenshot("../dataset/screenshots/example_1.png")
print(positions)
