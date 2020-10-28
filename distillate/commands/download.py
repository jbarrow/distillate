from . import cli

from google.cloud import storage
from google.api_core.exceptions import NotFound

import click
import os


model_files = [
    'config.json', 'pytorch_model.bin', 'special_tokens_map.json',
    'tokenizer_config.json', 'training_args.bin', 'vocab.txt'
]

@cli.command()
@click.argument('model', required=True)
def download(model: str):
    gs_client = storage.Client.create_anonymous_client()
    bucket = gs_client.bucket('distillate', user_project=None)

    os.mkdir(model)

    click.echo(f'Downloading model: {model}')
    for file in model_files:
        try:
            click.echo(f'Downloading file: {file}')
            bucket.blob(f'{model}/{file}').download_to_filename(f'{model}/{file}')
        except NotFound:
            click.echo(f'{model} not found. Exiting.')
            break
