"""
Command to visualize annotations.
"""
from . import cli
from pathlib import Path
from distillate.dataset import DistillateDataset

import click


@cli.command()
@click.argument('dataset', required=True, type=Path)
@click.argument('output', required=True, type=Path)
def visualize(dataset: Path, output: Path):
    click.echo(f'Visualizing annotations for dataset: {dataset}.')
    # load the dataset
    dataset = DistillateDataset.load(dataset)
    dataset.visualize_annotations(output)
