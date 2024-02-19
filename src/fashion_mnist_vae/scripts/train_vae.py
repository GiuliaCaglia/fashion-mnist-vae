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


class Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


def main(epochs: int, device: Literal["cpu", "cuda"], conditional: bool, debug: bool):
    # Get Data
    print("Getting Data...")
    x_train, y_train, x_test, y_test = (
        torch.tensor(i, dtype=torch.float32) for i in utils.get_data()
    )

    # Preprocess Data
    print("Preprocessing data...")
    x = torch.cat([x_train, x_test], dim=0).reshape(-1, 1, 28, 28) / 255.0
    x = x[:100] if debug else x
    if conditional:
        y = torch.cat([y_train, y_test], dim=0).reshape(-1, 1)
        y = y[:100] if debug else y
        dataset = Dataset(x.to(device), y.to(device))
        data_loader: data.DataLoader = data.DataLoader(
            dataset, shuffle=True, batch_size=128
        )

    else:
        data_loader: data.DataLoader = data.DataLoader(
            x.to(device), shuffle=True, batch_size=128
        )

    # Train Model
    print("Initializing model...")
    model = autoencoder.ConditionalVariationalAutoencoder().to(device) if conditional else autoencoder.VariationalAutoEncoder().to(device)
    losses = model.train(data_loader=data_loader, epochs=epochs)
    sampler = pyro.infer.Predictive(
        model=model.model, guide=model.guide, num_samples=100
    )
    if conditional:
        samples = sampler.forward(x[:25].to(device), y[:25].to(device))["latent_space"].mean(axis=0)
    else:
        samples = sampler.forward(x[:25].to(device))["latent_space"].mean(axis=0)
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
    parser.add_argument("-c", "--conditional", dest="conditional", default=False, action="store_true")
    parser.add_argument("--debug", dest="debug", default=False, action="store_true")

    kwargs = vars(parser.parse_args(sys.argv[1:]))

    main(**kwargs)
