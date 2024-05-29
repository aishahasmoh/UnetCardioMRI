import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import *

def plot_res(orig, masks, preds):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=7, figsize=(20, 20))
    for a in ax:
      a.grid(False)
      a.axis('off')
    # plot the original image, its masks, and the predicted masks
    ax[0].imshow(orig)
    ax[1].imshow(masks[0])
    ax[2].imshow(preds[0])
    ax[3].imshow(masks[1])
    ax[4].imshow(preds[1])
    ax[5].imshow(masks[2])
    ax[6].imshow(preds[2])

    # set the titles of the subplots
    ax[0].set_title("Original Image")
    ax[1].set_title("True Heart")
    ax[2].set_title("Predicted Heart")
    ax[3].set_title("True Left Ventricle")
    ax[4].set_title("Predicted Left Ventricle")
    ax[5].set_title("True Right Ventricle")
    ax[6].set_title("Predicted Right Ventricle")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()

def plot_res_compact(orig, mask, preds):
    # initialize our figure
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))
    for a in ax:
      a.grid(False)
      a.axis('off')
    # plot the original image, its masks, and the predicted masks
    ax[0].imshow(orig)
    ax[1].imshow(mask)
    ax[2].imshow(preds[0])
    ax[3].imshow(preds[1])
    ax[4].imshow(preds[2])

    # set the titles of the subplots
    ax[0].set_title("Original Image")
    ax[1].set_title("True Masks")
    ax[2].set_title("Predicted Heart")
    ax[3].set_title("Predicted Left Ventricle")
    ax[4].set_title("Predicted Right Ventricle")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()

def predict(loader, model):
    model.eval()
    with torch.no_grad():
        # loop over the validation set
        for (img, masks) in test_loader:
            # send the input to the device
            (x, y) = (img.to(DEVICE), masks.to(DEVICE))
            # make the predictions
            pred = model(x)
            pred = pred.squeeze()
            softmax = nn.Softmax(dim=1)
            softmax = torch.argmax(softmax(pred),axis=1).to('cpu')
            preds = torch.vsplit(pred[0,:,:,:], 3)
            blue_pred = np.transpose((preds[0].cpu().numpy()> THRESHOLD) * 225,  axes=[1,2,0])
            red_pred = np.transpose((preds[1].cpu().numpy()> THRESHOLD) * 225, axes=[1,2,0])
            green_pred = np.transpose((preds[2].cpu().numpy()> THRESHOLD) * 225, axes=[1,2,0])
            preds = (blue_pred.astype(np.uint8), red_pred.astype(np.uint8), green_pred.astype(np.uint8))
            x = np.transpose((x[0,:,:,:]).cpu().numpy(), axes=[1,2,0])
            y = np.transpose((y[0,:,:]).cpu().numpy(), axes=[1,2,0]).squeeze()
            plot_res_compact(x, y, preds)


# load saved model

# load test_loader
