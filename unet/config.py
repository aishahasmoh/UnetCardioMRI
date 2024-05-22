# imports and hyper-parameters
import torch
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

# base path of the dataset
LABELED_DATASET_PATH = os.path.join("images", "Traced")
OUTPUT_PATH = "output"
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join("images", "Untraced")
BLUE_MASK_DATASET_PATH = os.path.join("images", "blue")
RED_MASK_DATASET_PATH = os.path.join("images", "red")
GREEN_MASK_DATASET_PATH = os.path.join("images", "green")

os.makedirs(BLUE_MASK_DATASET_PATH, exist_ok=True)
os.makedirs(RED_MASK_DATASET_PATH, exist_ok=True)
os.makedirs(GREEN_MASK_DATASET_PATH, exist_ok=True)

TEST_RATIO = 0.1 # 90-10 split


# Model hyper-parameters
# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# Training hyper-parameters
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 8
# define the input image dimensions
INPUT_IMAGE_WIDTH = 224
INPUT_IMAGE_HEIGHT = 224
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(OUTPUT_PATH, "unet.pth")
PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "plot.png"])
TEST_PATHS = os.path.sep.join([OUTPUT_PATH, "test_paths.txt"])


# Setup device-agnostic code
if torch.cuda.is_available():
    DEVICE = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps" # Apple GPU
else:
    DEVICE = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
print(f"Using device: {DEVICE}")

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False