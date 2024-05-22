from imutils import paths
from sklearn.model_selection import train_test_split
from torchvision import transforms

from config import *
from dataset import CardiacMRIDataset

# I used cross entropy loss. I originally tried using Dice loss to try to account for the imbalance in image area occupied by heart vs not heart, but I did not get good results. I may try retraining with some sort of IoU-like metric so that the loss function takes into account more than just per-pixel error.


# load the image and mask filepaths in a sorted order
blue_mask_paths = sorted(list(paths.list_images(BLUE_MASK_DATASET_PATH)))
red_mask_paths = sorted(list(paths.list_images(RED_MASK_DATASET_PATH)))
green_mask_paths = sorted(list(paths.list_images(GREEN_MASK_DATASET_PATH)))
# only keep original images with matching maskss
image_paths = sorted([s.replace("green", "Untraced") for s in green_mask_paths])
#image_paths = sorted(list(set(paths.list_images(IMAGE_DATASET_PATH))))

# combine masks



split = train_test_split(image_paths, blue_mask_paths, red_mask_paths, green_mask_paths, test_size=TEST_RATIO, random_state=42)
train_images, test_images = split[:2]
b_train_masks, b_test_masks = split[2:4]
r_train_masks, r_test_masks = split[4:6]
g_train_masks, g_test_masks = split[6:8]


transformations = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((INPUT_IMAGE_HEIGHT,
		INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

train_dataset = CardiacMRIDataset(train_images, 
                                  (b_train_masks, r_train_masks, g_train_masks),
                                  transforms=transformations)
test_dataset = CardiacMRIDataset(test_images,(b_test_masks, r_test_masks,
                                  g_test_masks), transforms=transformations)

print(f"[INFO] found {len(train_dataset)} examples in the training set...")
print(f"[INFO] found {len(test_dataset)} examples in the test set...")