import cv2
import numpy as np
import os

def generate_heatmap(img_path):

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect edges
    edges = cv2.Canny(gray,50,150)

    # blur edges to create smooth heatmap
    heat = cv2.GaussianBlur(edges,(21,21),0)

    # normalize
    heat = cv2.normalize(heat,None,0,255,cv2.NORM_MINMAX)

    # apply color map
    heatmap = cv2.applyColorMap(heat,cv2.COLORMAP_JET)

    # overlay heatmap on original image
    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    filename = os.path.basename(img_path)

    heatmap_path = os.path.join("static/uploads","heatmap_"+filename)

    cv2.imwrite(heatmap_path,overlay)

    return heatmap_path