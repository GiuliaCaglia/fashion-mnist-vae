from typing import List

import pyro
import pyro.distributions as dist
import torch
from torch import nn, optim

from fashion_mnist_vae.networks import networks


class AutoEncoder(nn.Module):
    THICKNESS = 16
    LATENT_SPACE = 256

    def __init__(self):
        super().__init__()
        self.mainline = nn.Sequential(
            networks.Encoder(thickness=self.THICKNESS, latent_space=self.LATENT_SPACE),
            networks.Decoder(thickness=self.THICKNESS, latent_space=self.LATENT_SPACE),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)
        self.criterion = nn.MSELoss()

    def forward(self, X):
        return self.mainline(X)

    def train(self, data_loader, epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in data_loader:
                self.optimizer.zero_grad()
                X_hat = self.forward(batch)
                loss = self.criterion(X_hat, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss) / len(batch)
            losses.append(epoch_loss)
            print("Epoch: {}/{}; Loss: {}".format(epoch + 1, epochs, epoch_loss))

        return losses


class VariationalAutoEncoder(nn.Module):
    LATENT_SPACE = 256
    THICKNESS = 16

    def __init__(self):
        super().__init__()
        self.encoder = networks.Encoder(
            latent_space=self.LATENT_SPACE, thickness=self.THICKNESS, return_std=True
        )
        self.decoder = networks.Decoder(
            latent_space=self.LATENT_SPACE, thickness=self.THICKNESS
        )

    def guide(self, x):
        pyro.module("encoder", self.encoder)

        with pyro.plate("data", len(x), subsample_size=len(x)):
            z_loc, z_scale = self.encoder(x)
            pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))

    def model(self, x):
        pyro.module("decoder", self.decoder)
        z_loc = torch.zeros(len(x), self.LATENT_SPACE).to(self.decoder.mainline[1].weight.device.type)
        z_scale = torch.ones(len(x), self.LATENT_SPACE).to(self.decoder.mainline[1].weight.device.type)

        with pyro.plate("data", len(x), subsample_size=len(x)):
            z = pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))
            loc_out = self.decoder(z)
            pyro.sample("obs", dist.ContinuousBernoulli(probs=loc_out).to_event(3), obs=x)

            return loc_out

    def train(self, data_loader, epochs) -> List[float]:
        pyro.clear_param_store()
        model = self.model
        guide = self.guide
        criterion = pyro.infer.Trace_ELBO()
        optim = pyro.optim.Adam({"lr": 5e-4})
        svi = pyro.infer.SVI(model=model, guide=guide, loss=criterion, optim=optim)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in data_loader:
                loss = svi.step(batch)
                epoch_loss += loss / len(batch)
            print(f"Epoch: {epoch+1}/{epochs}; Loss: {epoch_loss}")
            losses.append(epoch_loss)

        return losses
