# Multi-Class Image Segmentation on Cardio MRI images

### Pre-processing
Pre-process the dataset generating binary masks from human expert annotations.
It creates 3 new folders of binary masks:
- `images/blue` for the full heart segmentation
- `images/green` for the left ventricle segmentation
- `images/red`for the right ventricle segmentation

Each folder contains around binary masks for around 800 images.
The original untraced images before generating masks are in `images/original`.

```python
# visialize some inputs and outputs from the train and test dataset
import matplotlib.pyplot as plt

def plot_image(orig, mask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    # plot the original image, its masks, and the predicted masks
    ax[0].imshow(orig)
    ax[1].imshow(mask)
    # set the titles of the subplots
    ax[0].set_title("Original Image")
    ax[1].set_title("Original Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


for x,y in train_dataset:
  print(x.shape, y.shape)
  plot_image(np.transpose(x, axes=[1,2,0]), y.squeeze())
  break;
for x,y in test_dataset:
  print(x.shape, y.shape)
```

###Model
Image Segmentation classifies each pixel into a binary class. Either its part of the class or not. Instance segmentaion classifies each pixel into an instane of the class.

Architecture: U-Net. U-Net is a neural network designed for biomedical image segmentation.

Unet is a series of encoder layers followed by decoder layers, and skip connections between them.

Encoder layers are a series of: (double convolution, max_pooling) layers.

Decoder layers are a series of (Teanspose/up convolution, double convolution) layers.

Skip steps concatentate each encoder layer to the corresponding decoder layer of the same size before applying the decoder double convolution to it. These skip connections help share localization information which is needed for image segmentation.

I do the following optimizations:
* I do batch normalization after each convolution in the between double convolutions.
* I do dropout in-between double convolutions to prevent overfitting. 

Our Unet model is initialized by the following Hyper-parameters: in_channels


* TODO: in_channels=3 because we assume inputs are colored images with 3 RGB channels.
* TODO: out_channels=1 because its a binary classification problem. The masks are binary.
* TODO: conv_sizes is a list of convolutions sizes used in the encoder levels and decoder levels. 


Input layer size: (batch_size, image_width, image_height, in_channels)
To create a virtual environment:
```sh
virtualenv -p python3 torch
```
To use the virtual environment:
```sh
source ~/torch/bin/activate
pip install -r requirements.txt
```
To load the dataset in `images` directory and create `output` directory for
saving the model:
```sh
unzip UKBB-CMR-images.zip -d images
mv images/Traced/SQ/Traced\ 4\ chamber/ images/Traced/SQ/Traced/
mv images/Traced/SQ/Untraced\ Copies/ images/Traced/SQ/Originals/
mv images/Traced/PI/Original\ Untraced/ images/Traced/PI/Originals/

rm -rf output
rm -rf images/blue
rm -rf images/green
rm -rf images/red
rm -rf images/original
rm -rf */.DS_Store

mkdir images/blue
mkdir images/green
mkdir images/red
mkdir images/original

mkdir output
```
To pre-process images in the dataset to generate 3 binary masks for each image.
The first mask will segment the whole heart. The second mask will segment the 
left ventricle and the third mask will segment the right ventricle.
```sh
python pre_process.py
```
An example of the original image and masks generated is shown here:
![content/sample_img_masks.png](content/sample_img_masks.png)
