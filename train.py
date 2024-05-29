from imutils import paths
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np

from config import *
from dataset import CardiacMRIDataset
from model import UNet


def train(loader, model, optimizer, loss_fn, scaler, log, test_loader):
    train_loss = 0.0
    for bidx, (data, targets) in enumerate(tqdm(loader)):
        model.train()
        data = data.to(device=DEVICE, dtype=torch.float32)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        targetd = targets.type(torch.long)

        # perform a forward pass and calculate the training loss
        with torch.cuda.amp.autocast():
          pred = model(data)
          loss = loss_fn(pred, targets)
          # add the loss to the total training loss so far
          train_loss += loss
        # backward
        # zero out previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


    test_loss = 0.0
    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in test_loader:
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            test_loss += loss_fn(pred, y)

    # update our training history
    log["train_loss"].append(train_loss.cpu().detach().numpy())
    log["test_loss"].append(test_loss.cpu().detach().numpy())

    return train_loss, test_loss




def main():
    # load the image and mask filepaths in a sorted order
    image_paths = sorted(list(set(paths.list_images(IMAGE_DATASET_PATH))))

    blue_mask_paths = sorted(list(paths.list_images(BLUE_MASK_DATASET_PATH)))
    red_mask_paths = sorted(list(paths.list_images(RED_MASK_DATASET_PATH)))
    green_mask_paths = sorted(list(paths.list_images(GREEN_MASK_DATASET_PATH)))

    print(f"Loaded {len(image_paths)} images, {len(blue_mask_paths)} blue masks, {len(red_mask_paths)} red masks, and {len(green_mask_paths)} green masks")

    # TODO: combine masks

    # splt into train and test datasets
    split = train_test_split(image_paths, blue_mask_paths, red_mask_paths,
        green_mask_paths, test_size=TEST_RATIO, random_state=42)
    train_images, test_images = split[:2]
    b_train_masks, b_test_masks = split[2:4]
    r_train_masks, r_test_masks = split[4:6]
    g_train_masks, g_test_masks = split[6:8]

    print(f"Training dataset: {len(train_images)} images, {len(b_train_masks)} blue masks, {len(r_train_masks)} red masks, and {len(g_train_masks)} green masks")
    print(f"Testing dataset: {len(test_images)} images, {len(b_test_masks)} blue masks, {len(r_test_masks)} red masks, and {len(g_test_masks)} green masks")

    # PyTorch expects the input image samples to be in PIL format.
    # Resize our images to a particular input dimension.
    transformations = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
                                                                                                transforms.ToTensor()])

    train_dataset = CardiacMRIDataset(train_images,
        (b_train_masks, r_train_masks, g_train_masks), transforms=transformations)
    test_dataset = CardiacMRIDataset(test_images,
        (b_test_masks, r_test_masks,g_test_masks), transforms=transformations)
    print(f"Successfully loaded training dataset of size {len(train_dataset)}")
    print(f"Successfully loaded testing dataset of size {len(test_dataset)}")

    # shuffle the dataset before iterating over it using the data loader.
    # batch size = 8 to avoid memory issues

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE,
                                pin_memory=PIN_MEMORY, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE,
                                pin_memory=PIN_MEMORY, num_workers=os.cpu_count())

    model = UNet().to(DEVICE)
    # TODO: chenge the loss function?
    # nn.CrossEntropyLoss is usually applied for multi class segmentation.
    # your target should be a LongTensor, should not have the channel dimension,
    # and should contain the class indices in [0, nb_classes-1].
    #oss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr = INIT_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    # create a log that will be used later for plotting results.
    log = {"train_loss": [], "test_loss": []}


    # training loop
    print("Training the model...")
    for epoch in range(NUM_EPOCHS):
        train_loss, test_loss = train(train_loader, model, optimizer, loss_fn, scaler, log, test_loader)
        print(f"EPOCH: {epoch + 1}/{NUM_EPOCHS} Train loss: {train_loss}, Test loss: {test_loss}")

    # plot the average training and test loss per epoch
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.array(log["train_loss"]) / (len(train_dataset) // BATCH_SIZE), label="train_loss")
    plt.plot(np.array(log["test_loss"]) / (len(test_dataset) // BATCH_SIZE), label="test_loss")
    plt.title("Training/Testing Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)
    # serialize the model to disk
    torch.save(model, MODEL_PATH)

    exit(0)


if __name__ == "__main__":
    main()   