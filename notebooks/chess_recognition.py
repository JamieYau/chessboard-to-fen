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
image_path = "../dataset/board.png"
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

# %%
