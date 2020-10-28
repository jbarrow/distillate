"""
Command to process annotations and pdfs into a single dataset.
"""
from . import cli
from distillate.utils import load_annotations, load_html, to_funsd_format, pdf_page_from_image
from distillate.layoutlm import convert, seg, main
from distillate.dataset import DistillateDataset
from argparse import Namespace
from pathlib import Path

import json
import click


def make_layoutlm_labels(labels: Path, output: Path) -> None:
    with labels.open() as fp, output.open('w') as fo:
        for line in fp:
            label = line.strip()
            for tag in ['B', 'I', 'E', 'S']:
                fo.write(f'{tag}-{label}\n')
        fo.write('O')

@cli.command()
@click.argument('dataset', required=True, type=Path)
@click.argument('model', required=True)
# , help='Name of the model to download'
def train(dataset: Path, model: str):
    dataset_path = dataset
    dataset = DistillateDataset.load(dataset)

    click.echo(f'Training model: {model}.')
    annotations = load_annotations(dataset._annotations_file)

    image_names = {}
    for image in dataset._images.glob('*.png'):
        image_names[pdf_page_from_image(image)] = image.stem

    pages = {}
    for doc in dataset._bounding_boxes.glob('*.html'):
        pages.update(load_html(doc))

    annotated_pages = to_funsd_format(annotations, pages)

    for page, form in annotated_pages.items():
        with (dataset._annotations / f'{image_names[page]}.json').open('w', encoding='utf8') as fp:
            json.dump(form, fp, ensure_ascii=False)

    llm = dataset_path / 'layoutlm'
    llm.mkdir(exist_ok=True)

    args = Namespace(
        data_dir=dataset._annotations,
        data_split='train',
        output_dir=llm,
        model_name_or_path='bert-base-uncased',
        max_len=510
    )

    make_layoutlm_labels(dataset._labels_file, dataset_path/'labels_layoutlm.txt')

    convert(args)
    seg(args)

    train_args = Namespace(
        data_dir=llm,
        model_type='layoutlm',
        model_name_or_path=model,
        do_lower_case=True,
        do_train=True,
        do_eval=False,
        do_predict=False,
        config_name='',
        tokenizer_name='',
        overwrite_output_dir=True,
        overwrite_cache=True,
        cache_dir='',
        max_seq_length=512,
        num_train_epochs=3.0,
        logging_steps=10,
        save_steps=-1,
        output_dir='trained',
        labels=(dataset_path/'labels_layoutlm.txt'),
        per_gpu_train_batch_size=16,
        per_gpu_eval_batch_size=16,
        fp16=False,
        evaluate_during_training=True,
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

    import shutil
    shutil.copy(str(dataset_path/'labels_layoutlm.txt'), str(Path('trained')/'labels_layoutlm.txt'))

    # TODO: generate a split and do a train/dev dataset
