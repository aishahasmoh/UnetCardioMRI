from torch.utils.data import Dataset
import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from torchvision import transforms
from config import *


# Input image size: (174, 208, 3)
# The network will take in images of 224x224.
# batch size of the dataloader = 8.
# The data was shuffled before batching.

class CardiacMRIDataset(Dataset):
	def __init__(self, image_paths, masks_paths, transforms=None):
		# store the image and mask filepaths, and transforms
		self.image_paths = image_paths
		self.blue_mask_paths = masks_paths[0]
		self.red_mask_paths = masks_paths[1]
		self.green_mask_paths = masks_paths[2]
		self.transforms = transforms


	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.image_paths)

	def __getitem__(self, idx):
		# grab the image path from the current index
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		img = cv2.imread(self.image_paths[idx])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		blue_mask = cv2.imread(self.blue_mask_paths[idx], 0)
		red_mask = cv2.imread(self.red_mask_paths[idx], 0)
		green_mask = cv2.imread(self.green_mask_paths[idx], 0)


		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			img = self.transforms(img)
			blue_mask = self.transforms(blue_mask)
			red_mask = self.transforms(red_mask)
			green_mask = self.transforms(green_mask)

    # TODO: return the image and 3 masks not only one.
		masks =  torch.cat([blue_mask, red_mask, green_mask], axis=0)
		# return a tuple of the image and its mask
		return img, masks