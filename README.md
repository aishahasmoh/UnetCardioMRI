### Pre-processing
Pre-process the dataset generating binary masks from human expert annotations.
It creates 3 new folders of binary masks:
- `images/blue` for the full heart segmentation
- `images/green` for the left ventricle segmentation
- `images/red`for the right ventricle segmentation

Each folder contains around binary masks for around 800 images.
The original untraced images before generating masks are in `images/original`.


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


```sh
virtualenv -p python3 torch
source ~/torch/bin/activate
pip install -r requirements.txt

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

python pre_process.py
```