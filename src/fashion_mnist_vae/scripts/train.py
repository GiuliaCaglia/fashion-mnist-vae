"""Basic Training script."""

import argparse
import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils import data

from fashion_mnist_vae.networks import autoencoder
from fashion_mnist_vae.utils import constants, utils


def main(
    device: str = "cpu", batch_size: int = 128, epochs: int = 50, debug: bool = False
):
    # Load Data
    print("Loading data...")
    x_train, _, x_test, _ = (
        torch.tensor(i, dtype=torch.float32) for i in utils.get_data()
    )

    # Preprocess Data
    print("Preprocessing data...")
    x = torch.cat([x_train, x_test], dim=0).reshape(-1, 1, 28, 28) / 255.0
    x = x[:100] if debug else x
    data_loader: data.DataLoader = data.DataLoader(
        x.to(device), shuffle=True, batch_size=batch_size
    )

    # Initialize Model
    print("Initializing model...")
    model = autoencoder.AutoEncoder().to(device)
    losses = model.train(data_loader=data_loader, epochs=epochs)

    # Predict first 25 Images and create image grid
    test_images = (
        model(x[:25].to(device)).cpu().detach().numpy().reshape(-1, 28, 28) * 255.0
    )
    image_grid = utils.image_grid(test_images, 5, 5)

    # Save assets
    constants.ASSETS_DIR.mkdir(exist_ok=True)
    plt.plot(losses)
    plt.savefig(constants.LOSS_PLOT.as_posix())
    torch.save(model, constants.MODEL)
    image_grid.save(constants.EXAMPLES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # prog="Autoencoder Training Script",
        description="Train Autoencoder",
        epilog="Thank you for using me!",
    )
    parser.add_argument(
        "-d", "--device", type=str, dest="device", action="store", default="cpu"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, dest="batch_size", action="store", default=128
    )
    parser.add_argument(
        "-e", "--epochs", type=int, dest="epochs", action="store", default=1
    )
    parser.add_argument("--debug", dest="debug", action="store_true", default=False)

    kwargs = vars(parser.parse_args(sys.argv[1:]))

    main(**kwargs)
