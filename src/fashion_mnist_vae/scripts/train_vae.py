"""Training script for VAE."""

import argparse
import sys
from typing import Literal

import dill
import matplotlib.pyplot as plt
import pyro.infer
import torch
from torch.utils import data
from torchvision.datasets import FashionMNIST

from fashion_mnist_vae.networks import autoencoder
from fashion_mnist_vae.utils import constants, utils


def main(epochs: int, device: Literal["cpu", "cuda"], conditional: bool):
    # Get Data
    print("Getting Data...")
    dataset = FashionMNIST(
        root=constants.ASSETS_DIR, transform=utils.to_tensor, download=True, train=True
    )
    samples_y = torch.tensor([dataset[i][1] for i in range(25)])
    samples_x = torch.stack([dataset[i][0] for i in range(25)], dim=0)

    # Preprocess Data
    print("Creating data loader...")
    data_loader: data.DataLoader = data.DataLoader(
        dataset, shuffle=True, batch_size=128
    )

    # Train Model
    print("Initializing model...")
    model = (
        autoencoder.ConditionalVariationalAutoencoder(device=device)
        if conditional
        else autoencoder.VariationalAutoEncoder(device=device)
    )
    losses = model.train(data_loader=data_loader, epochs=epochs)
    sampler = pyro.infer.Predictive(model=model.model, guide=model.guide, num_samples=1)
    if conditional:
        samples = sampler.forward(samples_x.to(device), samples_y.to(device))[
            "latent_space"
        ].mean(axis=0)
    else:
        samples = sampler.forward(samples_x.to(device))["latent_space"].mean(axis=0)
    decoded_samples = model.decoder(samples).reshape(25, 28, 28) * 255.0
    image_grid = utils.image_grid(decoded_samples.cpu().detach().numpy(), 5, 5)

    # Store Artifacts
    directory_name = "vae" + ("_con" if conditional else "")
    directory = constants.ASSETS_DIR.joinpath(directory_name)
    directory.mkdir(exist_ok=True, parents=True)
    plt.plot(losses)
    plt.savefig(directory.joinpath(constants.VAE_LOSS_PLOT).as_posix())
    with directory.joinpath(constants.MODEL).open("wb") as f:
        dill.dump(model, f)
    image_grid.save(directory.joinpath(constants.VAE_EXAMPLES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="VAE Training",
        description="Train a VAE Model",
        epilog="Thank you for training the VAE!",
    )
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=1)
    parser.add_argument("-d", "--device", dest="device", type=str, default="cpu")
    parser.add_argument(
        "-c", "--conditional", dest="conditional", default=False, action="store_true"
    )

    kwargs = vars(parser.parse_args(sys.argv[1:]))

    main(**kwargs)
