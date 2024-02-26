from typing import Tuple

import torch
from PIL import Image
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
