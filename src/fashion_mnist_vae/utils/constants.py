"""Constants module."""

import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = ROOT_DIR.joinpath("assets")
LOSS_PLOT = pathlib.Path("loss_plot.png")
EXAMPLES = pathlib.Path("examples.png")
MODEL = pathlib.Path("model.pkl")
VAE_LOSS_PLOT = pathlib.Path("loss_plot_vae.png")
VAE_EXAMPLES = pathlib.Path("examples_vae.png")
