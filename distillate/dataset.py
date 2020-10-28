"""
Manage the state for a dataset.
"""

import logging
import hashlib

from typing import List, Dict, Optional
from pathlib import Path
from distillate.utils import extract, rename, load_image_from_filename, \
        render_bounding_boxes, load_annotations, pdf_page_from_image
from tqdm import tqdm

BLOCKSIZE = 65536

def hash_pdf(pdf: Path) -> str:
    hasher = hashlib.md5()

    with pdf.open('rb') as fp:
        buf = fp.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = fp.read(BLOCKSIZE)

    return hasher.hexdigest()

class DistillateDataset(object):
    """
    A DistillateDataset is a wrapper around a folder structure for storing
    PDFs, images of the pages, bounding boxes, annotations, training data, etc.

    It also contains an index of the PDF files, which contains an MD5 hash
    of each file, so if additional PDFs are added we don't include duplicates.
    """

    _base: Path
    _index: Dict[str, str] = {}

    def __init__(self, base: Path,
                 pdfs: List[Path] = [],
                 labels: List[str] = [],
                 annotations: Optional[Path] = None) -> None:

        self._base = base
        self._pdfs = self._base / 'pdfs'
        self._images = self._base / 'images'
        self._bounding_boxes = self._base / 'bounding_boxes'
        self._annotations = self._base / 'annotations'
        self._annotations_file = self._base / 'labels.csv'
        self._index_file = self._base / 'index.tsv'
        self._labels_file = self._base / 'labels.txt'

        if not self._pdfs.exists():
            self._base.mkdir(parents=True, exist_ok=False)
            self._pdfs.mkdir()
            self._images.mkdir()
            self._bounding_boxes.mkdir()
            self._annotations.mkdir()
            self._index = {}
            self._labels = labels
            self.save_labels()
        else:
            self.load_index()
            self.load_labels()

        if pdfs is not None:
            self.add_pdfs(pdfs)

    @classmethod
    def load(cls, base: Path) -> 'DistillateDataset':
        return DistillateDataset(base)

    def add_pdfs(self, pdfs: List[Path]):
        for pdf in tqdm(pdfs):
            if not self.in_index(pdf):
                output = rename(pdf, copy=True, dest=self._pdfs)
                self.add_to_index(output)
                self.preprocess(output)
            else:
                logging.warn(f'Skipping {pdf}, as it has already been indexed')

    def save_labels(self) -> None:
        with self._labels_file.open('w') as fp:
            fp.write('\n'.join(self._labels))

    def load_labels(self) -> None:
        self._labels = []
        with self._labels_file.open() as fp:
            for line in fp:
                self._labels.append(line.strip())

    def load_index(self) -> None:
        self._index = {}
        with self._index_file.open() as fp:
            for line in fp:
                md5, doc = line.strip().split('\t')
                self._index[md5] = doc

    def in_index(self, pdf: Path) -> bool:
        pdf_hash = hash_pdf(pdf)
        return pdf_hash in self._index.keys()

    def add_to_index(self, pdf: Path):
        pdf_hash = hash_pdf(pdf)

        if pdf_hash not in self._index:
            self._index[pdf_hash] = str(pdf)
            with self._index_file.open('a') as fp:
                fp.write(f'{pdf_hash}\t{str(pdf)}\n')

    def preprocess(self, pdf: Path) -> None:
        extract(pdf, type='bounding_boxes', output_dir=self._bounding_boxes)
        extract(pdf, type='images', output_dir=self._images)

    def add_annotations(self, annotations: Path) -> None:
        with self._annotations_file.open('a') as fo, annotations.open() as fi:
            for line in fi:
                fo.write(line)

    def get_unannotated(self) -> List[Path]:
        return []

    def visualize_annotations(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True)

        annotations = load_annotations(self._annotations_file)

        for image in self._images.glob('*.png'):
            pdf_page = pdf_page_from_image(image)

            bounding_boxes = annotations.get(pdf_page, None)
            if bounding_boxes is None: continue

            im = load_image_from_filename(image)
            new_im = render_bounding_boxes(im, bounding_boxes)
            new_im.save(f'{str(output_dir)}/{image.stem}.png', 'PNG')
