import torch
from torch import nn


class Encoder(nn.Module):
    SHRINKAGE = 7

    def __init__(
        self,
        thickness: int,
        latent_space: int,
        return_std: bool = False,
        in_channel: int = 1,
    ):
        super().__init__()
        self.return_std = return_std
        self.mainline = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=thickness, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=thickness, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=thickness),
            nn.MaxPool2d(kernel_size=2),
            # Additional layer
            nn.Conv2d(
                in_channels=thickness, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=thickness, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=thickness, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=thickness),
            nn.MaxPool2d(kernel_size=2),
        )

        self.mean_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=thickness * self.SHRINKAGE**2, out_features=latent_space
            ),
            nn.LeakyReLU(),
        )
        self.std_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=thickness * self.SHRINKAGE**2, out_features=latent_space
            ),
        )

    def forward(self, X):
        out = X.clone()
        out = self.mainline(out)
        mean = self.mean_out(out)

        if self.return_std:
            std = torch.exp(self.std_out(out))
            return mean, std

        return mean


class Decoder(nn.Module):
    SHRINKAGE = 7

    def __init__(self, thickness: int, latent_space: int):
        super().__init__()
        self.thickness = thickness
        self.adapter = nn.Sequential(
            nn.Linear(
                in_features=latent_space, out_features=thickness * self.SHRINKAGE**2
            ),
            nn.LeakyReLU(),
        )
        self.mainline = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.BatchNorm2d(num_features=thickness),
            nn.ConvTranspose2d(
                in_channels=thickness, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=thickness, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=thickness, out_channels=thickness, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(),
            # Additional layer
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.BatchNorm2d(num_features=thickness),
            nn.ConvTranspose2d(
                in_channels=thickness, out_channels=thickness, kernel_size=5, padding=2
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=thickness, out_channels=thickness, kernel_size=5, padding=2
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=thickness, out_channels=1, kernel_size=3, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, X):
        out = X.clone()
        out = self.adapter(out).reshape(
            -1, self.thickness, self.SHRINKAGE, self.SHRINKAGE
        )
        out = self.mainline(out)
        return out


class NormalizingFlow(nn.Module):
    """Implements planar normalizing flow."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(1)
        self.activation = nn.Tanh()
        self.activation_derivative = lambda x: 1 - torch.pow(self.activation(x), 2)
        self.fc = nn.Linear(bias=True)

        self.layer = nn.Sequential(self.fc, self.activation)

    def forward(self, z):
        """Modify z as part of normalizing flow."""
        return z + self.scale * self.layer(z)


class LogDetJacobian(nn.Module):
    """Computes log-det Jacobian for normalizing flow."""

    def __init__(self, sister: NormalizingFlow) -> None:
        super().__init__()
        self.sister = sister

    def forward(self, z):
        """Calculate logdet-Jacobian for sister layer."""
        psi = self.sister.activation_derivative(self.sister.fc(z))
        jacobian = 1 + torch.matmul(self.sister.scale.transpose(), psi)

        return jacobian
