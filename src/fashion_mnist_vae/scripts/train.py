"""Basic Training script."""

import argparse
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from fashion_mnist_vae.networks import autoencoder
from fashion_mnist_vae.utils import constants, utils


def main(device: str = "cpu", batch_size: int = 128, epochs: int = 50):
    # Load Data
    print("Loading data...")
    dataset = FashionMNIST(
        root=constants.ASSETS_DIR, transform=utils.to_tensor, download=True, train=True
    )
    samples_x = torch.cat([dataset[i][0] for i in range(25)], dim=0)

    # Preprocess Data
    print("Preprocessing data...")
    data_loader: data.DataLoader = data.DataLoader(
        dataset=dataset, shuffle=True, batch_size=batch_size
    )

    # Initialize Model
    print("Initializing model...")
    model = autoencoder.AutoEncoder(device=device)
    losses = model.train(data_loader=data_loader, epochs=epochs)

    # Predict first 25 Images and create image grid
    test_images = (
        model(samples_x.to(device)).cpu().detach().numpy().reshape(-1, 28, 28) * 255.0
    )
    image_grid = utils.image_grid(test_images, 5, 5)

    # Save assets
    directory = constants.ASSETS_DIR.joinpath("autoencoder")
    directory.mkdir(exist_ok=True, parents=True)
    plt.plot(losses)
    plt.savefig(directory.joinpath(constants.LOSS_PLOT).as_posix())
    torch.save(model, directory.joinpath(constants.MODEL))
    image_grid.save(directory.joinpath(constants.EXAMPLES))


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

    kwargs = vars(parser.parse_args(sys.argv[1:]))

    main(**kwargs)
