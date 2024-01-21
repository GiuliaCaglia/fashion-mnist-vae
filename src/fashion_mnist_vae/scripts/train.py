"""Basic Training script."""

import argparse
import sys

import torch
from torch.utils import data

from fashion_mnist_vae.networks import autoencoder
from fashion_mnist_vae.utils import utils


def main(device: str = "cpu", batch_size: int = 128, epochs: int = 50):
    # Load Data
    x_train, _, x_test, _ = (
        torch.tensor(i, dtype=torch.float32) for i in utils.get_data()
    )

    # Preprocess Data
    x = torch.cat([x_train, x_test], dim=0).reshape(-1, 1, 28, 28) / 255.0
    data_loader: data.DataLoader = data.DataLoader(
        x.to(device), shuffle=True, batch_size=batch_size
    )

    # Initialize Model
    model = autoencoder.AutoEncoder().to(device)
    model.train(data_loader=data_loader, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Autoencoder Training Script",
        description="Train Autoencoder",
        epilog="Thank you for using me!",
    )
    parser.add_argument("-d", "--device", type=str, dest="device", action="store")
    parser.add_argument(
        "-b", "--batch-size", type=str, dest="batch_size", action="store"
    )
    parser.add_argument("-e", "--epochs", type=str, dest="epochs", action="store")

    kwargs = vars(parser.parse_args(sys.argv))

    main(**kwargs)
