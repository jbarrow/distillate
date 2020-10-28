"""
Command to inspect model outputs.
"""
from . import cli
from pathlib import Path
from distillate.utils import load_html, to_blank_funsd, pdf_page_from_image
from distillate.layoutlm import convert, seg, main
from distillate.dataset import DistillateDataset
from argparse import Namespace
from typing import List, Dict, Any

import json
import click


labels = [
    'HEADING',
    'OUTLINE',
    'PAGE_MARKER',
    'TEMPLATE',
    'SIGNATURE'
]

@cli.command()
@click.argument('model', required=True, type=str)
@click.argument('pdf', required=True, type=Path)
@click.option('--serialization-directory', type=Path, default=Path('./distillate_output'))
def predict(model: str, pdf: Path, serialization_directory: Path = Path('./distillate_output')):
    click.echo(f'Predicting using model: {model}.')

    # generate images/bounding boxes from pdf
    dataset = DistillateDataset(serialization_directory, pdfs=[pdf], labels=labels)

    bounding_boxes = list(dataset._bounding_boxes.glob('*.html'))[0]
    # generate funsd format
    pages = load_html(bounding_boxes)
    pages = to_blank_funsd(pages)

    image_names = {}
    for image in dataset._images.glob('*.png'):
        image_names[pdf_page_from_image(image)] = image.stem

    for page, form in pages.items():
        with (dataset._annotations / f'{image_names[page]}.json').open('w', encoding='utf8') as fp:
            json.dump(form, fp, ensure_ascii=False)

    llm = serialization_directory / 'layoutlm'
    llm.mkdir(exist_ok=True)

    args = Namespace(
        data_dir=(serialization_directory / 'annotations'),
        data_split='test',
        output_dir=(serialization_directory / 'layoutlm'),
        model_name_or_path='bert-base-uncased',
        max_len=510
    )

    convert(args)
    seg(args)

    train_args = Namespace(
        data_dir=(serialization_directory / 'layoutlm'),
        model_type='layoutlm',
        model_name_or_path=model,
        do_lower_case=True,
        do_train=False,
        do_eval=False,
        do_predict=True,
        config_name='',
        tokenizer_name='',
        overwrite_output_dir=True,
        overwrite_cache=True,
        cache_dir='',
        max_seq_length=512,
        num_train_epochs=-1,
        logging_steps=10,
        save_steps=-1,
        output_dir=model,
        labels=(Path(model)/'labels_layoutlm.txt'),
        per_gpu_train_batch_size=16,
        per_gpu_eval_batch_size=16,
        fp16=False,
        evaluate_during_training=False,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        max_steps=-1,
        warmup_steps=0,
        no_cuda=True,
        seed=42,
        fp16_opt_level="",
        local_rank=-1,
        server_ip='',
        server_port=''
    )

    main(train_args)
