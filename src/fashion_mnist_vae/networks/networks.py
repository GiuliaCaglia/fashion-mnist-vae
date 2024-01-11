import torch


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mainline = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.mean_out = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=8 * 14 * 14, out_features=128),
            torch.nn.ReLU(),
        )
        self.std_out = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=8 * 14 * 14, out_features=128),
        )

    def forward(self, X):
        out = X.clone()
        out = self.mainline(out)
        mean = self.mean_out(out)
        # std = torch.exp(self.std_out)

        return mean  # , std


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=8 * 14 * 14),
            torch.nn.ReLU(),
        )
        self.mainline = torch.nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(
                in_channels=8, out_channels=8, kernel_size=3, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=8, out_channels=8, kernel_size=3, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=8, out_channels=1, kernel_size=3, padding=1
            ),
            torch.nn.ReLU(),
        )

    def forward(self, X):
        out = X.clone()
        out = self.adapter(out).reshape(-1, 8, 14, 14)
        out = self.mainline(out)
        return out
