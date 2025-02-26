# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
image_path = "../dataset/video_1.png"
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
