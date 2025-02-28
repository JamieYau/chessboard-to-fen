import cv2 as cv

from fen_conversion import create_analysis_urls, create_fen
from image_processing import find_chessboard
from piece_detection import detect_pieces


def process_screenshot(image_path):
    screenshot = cv.imread(image_path)
    chessboard = find_chessboard(screenshot)
    piece_positions = detect_pieces(chessboard)
    fen = create_fen(piece_positions)
    lichess_url, chesscom_url = create_analysis_urls(fen)
    return fen, lichess_url, chesscom_url

fen, lichess_url, chesscom_url = process_screenshot("../dataset/screenshots/full_screenshot.png")
print(f"FEN: {fen}")
print(f"Lichess Analysis: {lichess_url}")
print(f"Chess.com Analysis: {chesscom_url}")