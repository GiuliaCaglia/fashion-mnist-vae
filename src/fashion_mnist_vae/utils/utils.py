from typing import List, Tuple

import torch
from PIL import Image
from pyro.infer import Trace_ELBO
from torchvision.transforms import ToTensor


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

    def loss(self, model, guide, jacobians: torch.tensor, *args, **kwargs):
        """Compute ELBO loss and add sum of jacobians along first axis."""
        trace_elbo_loss = super().loss(model, guide, *args, **kwargs)
        sum_jacobians = torch.sum(jacobians, axis=0)

        return trace_elbo_loss + sum_jacobians
