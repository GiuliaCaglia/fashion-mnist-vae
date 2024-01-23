"""Constants module."""

import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = ROOT_DIR.joinpath("assets")
LOSS_PLOT = ASSETS_DIR.joinpath("loss_plot.png")
EXAMPLES = ASSETS_DIR.joinpath("examples.png")
MODEL = ASSETS_DIR.joinpath("model.pkl")
VAE_MODEL = ASSETS_DIR.joinpath("model_vae.pkl")
VAE_LOSS_PLOT = ASSETS_DIR.joinpath("loss_plot_vae.png")
VAE_EXAMPLES = ASSETS_DIR.joinpath("examples_vae.png")
