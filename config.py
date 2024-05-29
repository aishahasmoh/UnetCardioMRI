import os
import torch



# base path of the dataset
LABELED_DATASET_PATH = os.path.join("images", "Traced")
OUTPUT_PATH = "output"
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join("images", "original")
BLUE_MASK_DATASET_PATH = os.path.join("images", "blue")
RED_MASK_DATASET_PATH = os.path.join("images", "red")
GREEN_MASK_DATASET_PATH = os.path.join("images", "green")

os.makedirs(BLUE_MASK_DATASET_PATH, exist_ok=True)
os.makedirs(RED_MASK_DATASET_PATH, exist_ok=True)
os.makedirs(GREEN_MASK_DATASET_PATH, exist_ok=True)
os.makedirs(IMAGE_DATASET_PATH, exist_ok=True)

# Dataset hyper-parameters
TEST_RATIO = 0.1 # 90-10 train-test split
# define the input image dimensions. resize input images and masks to this size
INPUT_IMAGE_WIDTH = 224
INPUT_IMAGE_HEIGHT = 224

# Model hyper-parameters
# define the number of channels in the input, number of classes,
# and sizes of layers in the U-Net model
NUM_CHANNELS = 3 # TODO: should it be 1?
NUM_CLASSES = 1 # TODO: should it be 4?
LAYER_SIZES = [64, 128, 256, 512] # TODO: mayeb start with 16?
CONV_K = 3 # convolution kernel size
POOL_K = 2 # max pooling kernel size
POOL_S = 2 # max pooling stride size
UP_CONV_K = 2 # size of up convoltuion kernel
UP_CONV_S = 2 # size of up convoltuion stride
DROPOUT = 0.1 # percentage of droppoed out pixels to prevent overfitting

# Training hyper-parameters
# initialize learning rate, number of training epochs , and batch size
INIT_LR = 0.001
NUM_EPOCHS = 20
BATCH_SIZE = 8


# define prediction threshold: probability a pixel bbelongs to a class to filter weak predictions
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

# create a log that will be used later for plotting results.
Log = {"train_loss": [], "test_loss": []}