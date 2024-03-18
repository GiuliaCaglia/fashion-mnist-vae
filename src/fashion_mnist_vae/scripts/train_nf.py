"""Train VAE with normalizing flow."""

from typing import Literal

import click
import dill
import matplotlib.pyplot as plt
import pyro
from torch import nn

from fashion_mnist_vae.networks import autoencoder
from fashion_mnist_vae.utils import constants, utils


@click.command()
@click.option("--epochs", "-e", default=100)
@click.option("--device", "-d", default="cpu")
@click.option("--flow-depth", "-f", default=1)
@click.option("--conditional", default=False, is_flag=True)
def train(epochs: int, device: Literal["cpu", "cuda"], flow_depth: int, conditional: bool = False):
    # Prepare Data
    dataset, samples_x, samples_y = utils.DataManager.get_data()
    data_loader = utils.DataManager.get_data_loader(dataset)

    # Train Model
    if conditional:
        model = autoencoder.CNFVAE(
            device=device, flow_lenght=flow_depth
        )
    else:
        model = autoencoder.NormalizingFlowAutoencoder(
            device=device, flow_lenght=flow_depth
        )
    losses = model.train(data_loader=data_loader, epochs=epochs)

    sampler = pyro.infer.Predictive(model=model.model, guide=model.guide, num_samples=1)
    samples = sampler.forward(samples_x.to(device), samples_y.to(device))["latent_space"].mean(axis=0)
    decoded_samples = model.decoder(samples).reshape(25, 28, 28) * 255.0
    image_grid = utils.image_grid(decoded_samples.cpu().detach().numpy(), 5, 5)

    # Store Artifacts
    directory_name = "vae_nf" if not conditional else "vae_con_nf"
    directory = constants.ASSETS_DIR.joinpath(directory_name)
    directory.mkdir(exist_ok=True, parents=True)
    plt.plot(losses)
    plt.savefig(directory.joinpath(constants.VAE_LOSS_PLOT).as_posix())
    with directory.joinpath(constants.MODEL).open("wb") as f:
        dill.dump(model, f)
    image_grid.save(directory.joinpath(constants.VAE_EXAMPLES))


if __name__ == "__main__":
    train()
