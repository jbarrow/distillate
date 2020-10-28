"""
Command to update a dataset with more PDFs OR more annotations (but not both).
"""
import click

from pathlib import Path
from typing import Optional, List
from distillate.dataset import DistillateDataset

from . import cli


@cli.command()
@click.argument('dataset', required=True, type=Path)
@click.option('--pdfs', type=Path, help='PDFs to include in the dataset')
@click.option('--annotations', type=Path, help='Annotations to include in the dataset')
def update(dataset: Path,
           pdfs: Optional[Path] = None,
           annotations: Optional[Path] = None) -> None:
    """
    Script to create a dataset from a collection of PDFs and annotations
    """
    assert pdfs is None or annotations is None
    # create the dataset
    dataset = DistillateDataset.load(base=dataset)

    if pdfs is not None:
        pdfs = pdfs.glob('*.pdf')
        dataset.add_pdfs(pdfs)
    elif annotations is not None:
        dataset.add_annotations(annotations)
