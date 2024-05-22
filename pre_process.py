import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from config import *
def generate_binary_masks(img, filename):
    # mapping from color to hsv range. from empirical analysis of traces.
    colors = {
        'blue': ((100,64,40), (140,255,255)),
        'green': ((38, 50, 50), (80, 255,255)),
        'red': ((0, 100, 20), (10, 255, 255), (160, 100, 20), (180, 255, 255)),
    }
    masks = []

    # convert image to hsv format
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color, range in sorted(colors.items()): # create blue, green, then red
        binay = cv2.inRange(hsv, range[0], range[1])
        if len(range) == 4:
            binay2 = cv2.inRange(hsv, range[2], range[3])
            binay = binay + binay2
        # create a black image, find countours, draw them
        mask = cv2.threshold(binay, 0, 0, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(binay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)

        #cv2.imshow('image', mask)
        cv2.imwrite(os.path.join("images", color, filename), mask)
        masks.append(mask)
    return masks



def preprocess_images(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            # generate a mask of blue, green, and red masks
            masks = generate_binary_masks(img, filename)
    return

images = []
for folder in os.listdir(LABELED_DATASET_PATH):
    preprocess_images(os.path.join(LABELED_DATASET_PATH, folder, "Traced"))