from imutils import paths
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

from config import *
from dataset import CardiacMRIDataset
from model import UNet


def train(loader, model, optimizer, loss_fn, scalar):
    train_loss = 0.0
    for bidx, (data, targets) in enumerate(tqdm(loader)):
        model.train()
        data = data.to(device=DEVICE)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # perform a forward pass and calculate the training loss
        pred = model(data)

        #print(f"x size = {data.size()} y size = {targets.size()}, pred = {pred.size()}")

        loss = loss_fn(pred, targets)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # add the loss to the total training loss so far
        train_loss += loss

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

    # calculate the average training and validation loss
    train_loss = train_loss / (len(train_dataset) // BATCH_SIZE)
    test_loss = test_loss / (len(test_dataset) // BATCH_SIZE)

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
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = INIT_LR)
    # create a log that will be used later for plotting results.
    log = {"train_loss": [], "test_loss": []}

    # training loop
    print("Training the model...")
    for epoch in range(NUM_EPOCHS):
        epoch_loss, test_loss = train(train_loader, model, optimizer, loss_fn, 0)
        print(f"EPOCH: {epoch + 1}/{NUM_EPOCHS} Train loss: {epoch_loss:.6f}, Test loss: {test_loss:.4f}")


    # plot the training and test loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(log["train_loss"], label="train_loss")
    plt.plot(log["test_loss"], label="test_loss")
    plt.title("Training/Testing Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)
    # serialize the model to disk
    torch.save(model, MODEL_PATH)

if __name__ == "__main__":
    main()   