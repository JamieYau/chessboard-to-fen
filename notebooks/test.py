# %%
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# %%

def resize_image(image, scale):
    height, width = image.shape[:2]
    heightScale = int(height * scale)
    widthScale = int(width * scale)
    return cv.resize(image, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)

def show_image(image, title="Image"):
    while True:
        if cv.waitKey(1) == ord('q'):
            break
        cv.imshow(title, image)
    cv.destroyWindow(title)

# Read image
root = os.getcwd()
IMAGE_PATH = os.path.join(root, "../dataset/screenshots/board.png")
IMG_BGR = cv.imread(IMAGE_PATH)
IMG_BGR = resize_image(IMG_BGR, 8/10)
IMG_RGB = cv.cvtColor(IMG_BGR, cv.COLOR_BGR2RGB)
IMG_GRAY = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)
IMG_GRAY = resize_image(IMG_GRAY, 8/10)
IMG_BLUR = cv.GaussianBlur(IMG_GRAY, (5, 5), 0)
IMG_CLAHE = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
IMG_CLAHE = IMG_CLAHE.apply(IMG_GRAY)

# %%

def global_threshold(image):
    _, thresh = cv.threshold(image, 80, 160, cv.THRESH_BINARY)
    return thresh

def adaptive_threshold(image):
    thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return thresh

def otsu_threshold(image):
    _, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

def canny_edge_visualization(image):
    img = image
    winname = "Canny Edge Detection"
    cv.namedWindow(winname)
    cv.createTrackbar("minThresh", winname, 0, 255, lambda x: None)
    cv.createTrackbar("maxThresh", winname, 0, 255, lambda x: None)
    
    while True:
        if cv.waitKey(1) == ord('q'):
            break
        
        minThresh = cv.getTrackbarPos("minThresh", winname)
        maxThresh = cv.getTrackbarPos("maxThresh", winname)
        cannyEdges = cv.Canny(img, minThresh, maxThresh)
        cv.imshow(winname, cannyEdges)
        
    cv.destroyWindow(winname)

def canny_edge_detection(image):
    return cv.Canny(image, 10, 20)

# %%

canny_edge_visualization(IMG_GRAY)
canny_edges = canny_edge_detection(IMG_BLUR)
show_image(canny_edges, "Canny Edges")

# %%

def find_contours(canny_edges):
    contours, _ = cv.findContours(canny_edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def largest_contour(contours):
    # Filter by area and approximate shape  
    board_contour = None  
    for cnt in contours:  
        area = cv.contourArea(cnt)  
        peri = cv.arcLength(cnt, True)  
        approx = cv.approxPolyDP(cnt, 0.02*peri, True)  
        # Look for the largest quadrilateral (4 corners)  
        if len(approx) == 4 and area > 100:  # Adjust area threshold  
            board_contour = approx  
            break
    return board_contour

def draw_contours(image, contours):
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    show_image(image, "Contour")

# %%

contours = find_contours(canny_edges)
largest_contour = largest_contour(contours)
IMG_RGB_COPY = IMG_RGB.copy()
draw_contours(IMG_RGB_COPY, largest_contour)
IMG_RGB_COPY_2 = IMG_RGB.copy()

def find_lines(canny_edges):
    lines = cv.HoughLinesP(canny_edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=50)
    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) == 90:  # Vertical
            vertical.append(line)
        elif abs(angle) == 0:  # Horizontal
            horizontal.append(line)
            
    for line in horizontal:
        x1, y1, x2, y2 = line[0]
        cv.line(IMG_RGB_COPY_2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.circle(IMG_RGB_COPY_2, (x1, y1), 5, (0, 255, 0), -1)
        cv.circle(IMG_RGB_COPY_2, (x2, y2), 5, (255, 0, 0), -1)
        
    for line in vertical:
        x1, y1, x2, y2 = line[0]
        cv.line(IMG_RGB_COPY_2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.circle(IMG_RGB_COPY_2, (x1, y1), 5, (0, 255, 0), -1)
        cv.circle(IMG_RGB_COPY_2, (x2, y2), 5, (255, 0, 0), -1)
        
    show_image(IMG_RGB_COPY_2, "Lines")
    
    return horizontal, vertical

horizontal, vertical = find_lines(canny_edges)

# %%
def template_matching(image, template):
    template = resize_image(template, 1)
    height, width = template.shape[:2]
    templateMap = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
    _, _, min_loc, max_loc = cv.minMaxLoc(templateMap)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    
    show_image(image, "Template Matching")
    
TEMPLATE_PATH = os.path.join(root, "../dataset/screenshots/template.png")
template = cv.imread(TEMPLATE_PATH, cv.IMREAD_GRAYSCALE)
IMG_RGB_COPY_3 = IMG_GRAY.copy()
template_matching(IMG_RGB_COPY_3, template)

# %%

def test(image):
    cv.imshow("Image", image)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)

    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    cv.imshow("thresh", thresh)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    c = 0
    for i in contours:
            area = cv.contourArea(i)
            if area > 1000:
                    if area > max_area:
                        max_area = area
                        best_cnt = i
                        image = cv.drawContours(image, contours, c, (0, 255, 0), 3)
            c+=1

    mask = np.zeros((gray.shape),np.uint8)
    cv.drawContours(mask,[best_cnt],0,255,-1)
    cv.drawContours(mask,[best_cnt],0,0,2)
    cv.imshow("mask", mask)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]
    cv.imshow("New image", out)

    blur = cv.GaussianBlur(out, (5,5), 0)
    cv.imshow("blur1", blur)

    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    cv.imshow("thresh1", thresh)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    c = 0
    for i in contours:
            area = cv.contourArea(i)
            if area > 1000/2:
                cv.drawContours(image, contours, c, (0, 255, 0), 3)
            c+=1


    cv.imshow("Final Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
#test(IMG_BGR)

# %%
