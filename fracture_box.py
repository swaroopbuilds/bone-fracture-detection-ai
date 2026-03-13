import cv2
import os

def detect_fracture_box(img_path):

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Edge detection
    edges = cv2.Canny(blur,50,150)

    # Find contours
    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img_path

    # Find largest contour
    largest_contour = max(contours,key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(largest_contour)

    # Draw rectangle
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    box_path = img_path.replace(".", "_box.")

    cv2.imwrite(box_path,img)

    return box_path