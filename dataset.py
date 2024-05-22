# import the necessary packages
from torch.utils.data import Dataset
import cv2


# Input image size: (174, 208, 3)
# The network will take in images of 224x224.
# batch size of the dataloader = 8.
# The data was shuffled before batching.


class CardiacMRIDataset(Dataset):
	def __init__(self, image_paths, masks_paths, transforms=None):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_paths = image_paths
		self.blue_mask_paths = masks_paths[0]
		self.red_mask_paths = masks_paths[1]
		self.green_mask_paths = [2]
		self.transforms = transforms


	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		# grab the image path from the current index
		img = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		img = cv2.imread(self.image_paths[idx])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.blue_mask_paths[idx], 0)
		mask = cv2.imread(self.red_mask_paths[idx], 0)
		mask = cv2.imread(self.green_mask_paths[idx], 0)
		mask[mask == 255] = 1.0 # because predictions are output of softmax


		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			img = self.transforms(img)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return img, mask

