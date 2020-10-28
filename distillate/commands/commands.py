"""
Command to download a PDF model.

$ distillate download [MODEL]
"""

from . import cli

from .create import create
from .update import update
from .visualize import visualize
from .train import train
from .download import download
from .predict import predict

# TODO: write command to check if Poppler (or other PDF tools) are installed

if __name__ == '__main__':
    cli()
