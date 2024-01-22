import pyro
import pyro.distributions as dist
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
    def __init__(self):
        super().__init__()
        self.encoder = networks.Encoder()
        self.decoder = networks.Decoder()

    def guide(self, X):
        pyro.module("encoder", self.encoder)

        with pyro.plate("data", len(X)):
            z_loc, z_scale = self.encoder(X)
            pyro.sample("latent_space", dist.Normal(z_loc, z_scale).to_event(1))
