from typing import List, Literal

import pyro
import pyro.distributions as dist
import torch
from torch import nn, optim

from fashion_mnist_vae.networks import networks
from fashion_mnist_vae.utils import utils


class AutoEncoder(nn.Module):
    THICKNESS = 16
    LATENT_SPACE = 256

    def __init__(self, device: Literal["cpu", "cuda"] = "cpu"):
        super().__init__()
        self.mainline = nn.Sequential(
            networks.Encoder(thickness=self.THICKNESS, latent_space=self.LATENT_SPACE),
            networks.Decoder(thickness=self.THICKNESS, latent_space=self.LATENT_SPACE),
        ).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)
        self.criterion = nn.MSELoss()
        self.device = device

    def forward(self, X):
        return self.mainline(X.to(self.device))

    def train(self, data_loader, epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for x, _ in data_loader:
                self.optimizer.zero_grad()
                X_hat = self.forward(x.to(self.device))
                loss = self.criterion(X_hat, x.to(self.device))
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss) / len(x)
            losses.append(epoch_loss)
            print("Epoch: {}/{}; Loss: {}".format(epoch + 1, epochs, epoch_loss))

        return losses


class VariationalAutoEncoder(nn.Module):
    LATENT_SPACE = 256
    THICKNESS = 16

    def __init__(self, device: Literal["cpu", "cuda"] = "cpu"):
        super().__init__()
        self.encoder = networks.Encoder(
            latent_space=self.LATENT_SPACE, thickness=self.THICKNESS, return_std=True
        ).to(device)
        self.decoder = networks.Decoder(
            latent_space=self.LATENT_SPACE, thickness=self.THICKNESS
        ).to(device)
        self.device = device

    def guide(self, x):
        x = x.to(self.device)
        pyro.module("encoder", self.encoder)

        with pyro.plate("data", len(x), subsample_size=len(x)):
            z_loc, z_scale = self.encoder(x)
            pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))

    def model(self, x):
        x = x.to(self.device)
        pyro.module("decoder", self.decoder)

        with pyro.plate("data", len(x), subsample_size=len(x)):
            z_loc = torch.zeros(len(x), self.LATENT_SPACE).to(self.device)
            z_scale = torch.ones(len(x), self.LATENT_SPACE).to(self.device)
            z = pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))
            z_prime = self.transform_z(z)
            loc_out = self.decoder(z_prime)
            pyro.sample(
                "obs", dist.ContinuousBernoulli(probs=loc_out).to_event(3), obs=x
            )

        return loc_out

    def train(self, data_loader, epochs) -> List[float]:
        pyro.clear_param_store()
        model = self.model
        guide = self.guide
        criterion = pyro.infer.Trace_ELBO()
        optim = pyro.optim.ClippedAdam({"lr": 5e-4, "weight_decay": 1})
        svi = pyro.infer.SVI(model=model, guide=guide, loss=criterion, optim=optim)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for x, _ in data_loader:
                loss = svi.step(x.to(self.device))
                epoch_loss += loss / len(x)
            print(f"Epoch: {epoch+1}/{epochs}; Loss: {epoch_loss}")
            losses.append(epoch_loss)

        return losses

    def transform_z(self, z):
        return z


class ConditionalVariationalAutoencoder(VariationalAutoEncoder):
    def __init__(self, device: Literal["cpu", "cuda"] = "cpu"):
        super().__init__(device=device)
        self.encoder = networks.Encoder(
            thickness=self.THICKNESS,
            latent_space=self.LATENT_SPACE,
            return_std=True,
            in_channel=2,
        ).to(device)
        self.prior_net = networks.Encoder(
            thickness=self.THICKNESS,
            latent_space=self.LATENT_SPACE,
            return_std=True,
            in_channel=2,
        ).to(device)

    def model(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        pyro.module("decoder", self.decoder)
        pyro.module("prior_net", self.prior_net)

        z_base = y.reshape(-1, 1, 1, 1) * torch.ones_like(x)
        z_base = torch.cat([x, z_base], dim=1)
        z_loc, z_scale = self.prior_net(z_base)

        with pyro.plate("data", len(x), subsample_size=len(x)):
            z = pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))
            loc_out = self.decoder(z)
            pyro.sample(
                "obs", dist.ContinuousBernoulli(probs=loc_out).to_event(3), obs=x
            )

        return loc_out

    def guide(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        pyro.module("encoder", self.encoder)
        xy = torch.cat([x, y.reshape(-1, 1, 1, 1) * torch.ones_like(x)], dim=1)

        with pyro.plate("data", len(x), subsample_size=len(x)):
            z_loc, z_scale = self.encoder(xy)  # In the documentation there
            # is a case distinction between y being passed and not -> For our purposes, we can assume y to always be
            # given because we want to generate a picture of type y
            pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        xy = torch.cat([x, y.reshape(-1, 1, 1, 1) * torch.ones_like(x)], dim=1)
        compressed_mean, _ = self.encoder(xy)
        reconstructed = self.decoder(compressed_mean)

        return reconstructed

    def train(self, data_loader, epochs) -> List[float]:
        pyro.clear_param_store()
        model = self.model
        guide = self.guide
        criterion = pyro.infer.Trace_ELBO()
        optim = pyro.optim.ClippedAdam({"lr": 5e-4, "weight_decay": 1})
        svi = pyro.infer.SVI(model=model, guide=guide, optim=optim, loss=criterion)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in data_loader:
                loss = svi.step(x.to(self.device), y.to(self.device))
                epoch_loss += float(loss) / len(x)
            print(f"Epoch: {epoch+1}/{epochs}; Loss: {epoch_loss}")
            losses.append(epoch_loss)

        return losses


class NormalizingFlowAutoencoder(VariationalAutoEncoder):
    """Special VAE implementing normalizing flows.

    Normalizing flows have been suggested by Rezende & Mohamed (2015):
    http://proceedings.mlr.press/v37/rezende15.pdf
    """

    JACOBIAN = networks.LogDetJacobian

    def __init__(self, device: Literal["cpu", "cuda"] = "cpu"):
        super().__init__(device)
        self.normalizing_flows: nn.Sequential

    def add_normalizing_flows(
        self,
        flows: nn.Sequential,
    ):
        self.normalizing_flows = flows

    def transform_z(self, z):
        z_out = z.clone()
        log_det_jacobians = torch.zeros(z_out.shape[0], 1)
        try:
            for flow in self.normalizing_flows:
                z_out = flow(z_out)
                jacobian = self.JACOBIAN(flow)(z_out)
                log_det_jacobians += jacobian
        except AttributeError as e:
            raise AttributeError("No normalizing flows added!!") from e

        pyro.factor("log_det_jacobians", log_factor=log_det_jacobians * -1)
        return z_out
