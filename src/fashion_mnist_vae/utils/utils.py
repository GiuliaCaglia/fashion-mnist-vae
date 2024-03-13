from typing import List, Tuple

import torch
from PIL import Image
from pyro.infer import Trace_ELBO
from torch.utils import data
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from fashion_mnist_vae.utils import constants


def image_grid(imgs, rows, cols, gridsize: Tuple[int, int] = (200, 200)):
    assert len(imgs) == rows * cols

    w, h = imgs[0].shape
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        image = Image.fromarray(img)
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid.resize(gridsize)


def to_tensor(image: Image) -> torch.tensor:
    transform = ToTensor()
    tensor = transform(image).reshape(1, 28, 28)

    return tensor


class FreeEnergyLoss(Trace_ELBO):
    """Modification of Trace_ELBO to include sum of logdet-Jacobians."""

    def loss_and_grads(self, model, guide, jacobians: torch.tensor, *args, **kwargs):
        """Compute ELBO loss and add sum of jacobians along first axis."""
        trace_elbo_loss = super().loss_and_grads(model, guide, *args, **kwargs)
        sum_jacobians = torch.sum(jacobians, axis=0)

        return trace_elbo_loss + sum_jacobians


class DataManager:
    @staticmethod
    def get_data():
        # Get Data
        print("Getting Data...")
        dataset = FashionMNIST(
            root=constants.ASSETS_DIR, transform=to_tensor, download=True, train=True
        )
        samples_y = torch.tensor([dataset[i][1] for i in range(25)])
        samples_x = torch.stack([dataset[i][0] for i in range(25)], dim=0)

        return dataset, samples_x, samples_y

    @staticmethod
    def get_data_loader(dataset):
        # Preprocess Data
        print("Creating data loader...")
        data_loader: data.DataLoader = data.DataLoader(
            dataset, shuffle=True, batch_size=128
        )

        return data_loader
