"""
Command to create a new dataset.
"""
import click

from pathlib import Path
from typing import Optional, List
from distillate.dataset import DistillateDataset

from . import cli


@cli.command()
@click.argument('dataset', required=True, type=Path)
@click.option('--labels', type=str, help='Comma separated list of labels to use for annotation/training', required=True)
@click.option('--pdfs', type=Path, help='PDFs to include in the dataset')
def create(dataset: Path,
           labels: str,
           pdfs: Optional[Path] = None) -> None:
    """
    Script to create a dataset from a collection of PDFs and annotations
    """
    # glob the pdfs
    pdfs = pdfs.glob('*.pdf')
    # split the labels
    labels = labels.split(',')
    # create the dataset
    dataset = DistillateDataset(base=dataset, pdfs=pdfs, labels=labels)
