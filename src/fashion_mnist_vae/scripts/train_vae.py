"""Training script for VAE."""

import argparse
import sys
from typing import Literal

import matplotlib.pyplot as plt
import pyro.infer
import torch
from torch.utils import data

from fashion_mnist_vae.networks import autoencoder
from fashion_mnist_vae.utils import constants, utils


def main(epochs: int, device: Literal["cpu", "cuda"], debug: bool):
    # Get Data
    print("Getting Data...")
    x_train, _, x_test, _ = (
        torch.tensor(i, dtype=torch.float32) for i in utils.get_data()
    )

    # Preprocess Data
    print("Preprocessing data...")
    x = torch.cat([x_train, x_test], dim=0).reshape(-1, 1, 28, 28) / 255.0
    x = x[:100] if debug else x
    data_loader: data.DataLoader = data.DataLoader(
        x.to(device), shuffle=True, batch_size=128
    )

    # Train Model
    print("Initializing model...")
    model = autoencoder.VariationalAutoEncoder().to(device)
    losses = model.train(data_loader=data_loader, epochs=epochs)
    sampler = pyro.infer.Predictive(
        model=model.model, guide=model.guide, num_samples=100
    )
    samples = sampler.forward(x[:25])["latent_space"].mean(axis=0)
    decoded_samples = model.decoder(samples).reshape(25, 28, 28) * 255.0
    image_grid = utils.image_grid(decoded_samples.cpu().detach().numpy(), 5, 5)

    # Store Artifacts
    constants.ASSETS_DIR.mkdir(exist_ok=True)
    plt.plot(losses)
    plt.savefig(constants.VAE_LOSS_PLOT.as_posix())
    image_grid.save(constants.VAE_EXAMPLES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="VAE Training",
        description="Train a VAE Model",
        epilog="Thank you for training the VAE!",
    )
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=1)
    parser.add_argument("-d", "--device", dest="device", type=str, default="cpu")
    parser.add_argument("--debug", dest="debug", default=False, action="store_true")

    kwargs = vars(parser.parse_args(sys.argv[1:]))

    main(**kwargs)
