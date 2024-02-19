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
        z_loc = torch.zeros(len(x), self.LATENT_SPACE).to(
            self.decoder.mainline[1].weight.device.type
        )
        z_scale = torch.ones(len(x), self.LATENT_SPACE).to(
            self.decoder.mainline[1].weight.device.type
        )

        with pyro.plate("data", len(x), subsample_size=len(x)):
            z = pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))
            loc_out = self.decoder(z)
            pyro.sample(
                "obs", dist.ContinuousBernoulli(probs=loc_out).to_event(3), obs=x
            )

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


class ConditionalVariationalAutoencoder(VariationalAutoEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = networks.Encoder(thickness=self.THICKNESS, latent_space=self.LATENT_SPACE, return_std=True,
                                          in_channel=2)
        self.prior_net = networks.Encoder(thickness=self.THICKNESS, latent_space=self.LATENT_SPACE, return_std=True, in_channel=2)

    def model(self, x, y):
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
        pyro.module("encoder", self.encoder)
        xy = torch.cat([x, y.reshape(-1, 1, 1, 1) * torch.ones_like(x)], dim=1)

        with pyro.plate("data", len(x), subsample_size=len(x)):
            z_loc, z_scale = self.encoder(xy)  # In the documentation there
            # is a case distinction between y being passed and not -> For our purposes, we can assume y to always be
            # given because we want to generate a picture of type y
            pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))

    def forward(self, x, y):
        xy = torch.cat([x, y.reshape(-1, 1, 1, 1) * torch.ones_like(x)], dim=1)
        compressed_mean, _ = self.encoder(xy)
        reconstructed = self.decoder(compressed_mean)

        return reconstructed

    def train(self, data_loader, epochs) -> List[float]:
        pyro.clear_param_store()
        model = self.model
        guide = self.guide
        criterion = pyro.infer.Trace_ELBO()
        # elbo = lambda m, g, x, y: pyro.infer.Trace_ELBO().differentiable_loss(m, g, x, y)
        # ce = torch.nn.CrossEntropyLoss()
        # with pyro.poutine.trace(param_only=True) as param_capture:
        #     x, y = [(a, b) for a, b in data_loader][0]
        #     _ = elbo(model, guide, x, y)
        #     params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        # optimizer = torch.optim.Adam(params, lr=5e-4, betas=(0.90, 0.999))
        optim = pyro.optim.Adam({"lr": 5e-4})
        svi = pyro.infer.SVI(model=model, guide=guide, optim=optim, loss=criterion)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in data_loader:
                # optimizer.zero_grad()
                # xy = torch.cat([x, y.reshape(-1, 1, 1, 1) * torch.ones_like(x)], dim=1)
                # loss = elbo(model, guide, x, y) #+ ce(self.forward(x, y), x)
                # loss.backward()
                # optimizer.step()
                loss = svi.step(x, y)
                epoch_loss += float(loss) / len(x)
            print(f"Epoch: {epoch+1}/{epochs}; Loss: {epoch_loss}")
            losses.append(epoch_loss)

        return losses
