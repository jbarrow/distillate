import subprocess
import itertools
import shutil
import uuid
import json

from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Any, NamedTuple, Optional
from collections import defaultdict
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup


class DataType(Enum):
    text = 1
    bounding_boxes = 2
    images = 3


class BoundingBox(NamedTuple):
    """
    The (x1, y1) position is at the top left corner,
    the (x2, y2) position is at the bottom right corner
    """
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ''


class PDFPage(NamedTuple):
    filename: str
    page_num: int


def extract(path: Path, type: str = 'text', output_dir: Optional[Path] = None) -> Path:
    """
    Extract the contents of a PDF file. The current implementation
    simply grabs all the text, but in the future there are smarter layout
    processing things to be done.
    """

    data_type = DataType[type]

    if output_dir is None:
        output_dir = path.parent
    output_file = output_dir / path.name

    if data_type == DataType.bounding_boxes:
        # then run to generate html layout
        output = output_file.with_suffix('.html')
        subprocess.run(['pdftotext', '-bbox', str(path), str(output)])
    elif data_type == DataType.text:
        # run the version that generates the text only
        output = output_file.with_suffix('.txt')
        subprocess.run(['pdftotext', str(path), str(output)])
    elif data_type == DataType.images:
        output = output_file.with_suffix('')
        subprocess.run(['pdftoppm', '-png', str(path), str(output)])

    return output


def rename(path: Path, copy: bool = False, dest: Path = None) -> Path:
    """
    Rename a PDF file with a unique, non-recognizable ID.
    """
    unique_filename = str(uuid.uuid4())

    if not copy:
        renamed = path.parent / (unique_filename + path.suffix)
        shutil.move(str(path), str(renamed))
    else:
        renamed = dest / (unique_filename + path.suffix)
        shutil.copy(str(path), str(renamed))

    return renamed

def load_html_page(page: BeautifulSoup,
                   rescale_size: Tuple[float, float]) -> List[BoundingBox]:
    """
    Load the HTML bounding box output of pdftotext and convert it to a
    dict that contains all the bounding boxes.
    """
    boxes = []

    target_width, target_height = rescale_size
    current_width, current_height = float(page['width']), float(page['height'])
    rescale_width, rescale_height = target_width / current_width, target_height / current_height

    for word in page.find_all('word'):
        boxes.append(BoundingBox(**{
            'x1': float(word['xmin']) * rescale_width,
            'y1': float(word['ymin']) * rescale_height,
            'x2': float(word['xmax']) * rescale_width,
            'y2': float(word['ymax']) * rescale_height,
            'label': word.text
        }))

    return boxes


def load_html(path: Path) -> Dict[PDFPage, List[BoundingBox]]:
    pages = {}
    with path.open() as fp:
        soup = BeautifulSoup(fp.read(), 'html.parser')

        for i, page in enumerate(soup.find_all('page')):
            key = PDFPage(path.stem, i+1)
            rescale_size = load_image(path.parent.parent / 'images', key).size
            pages[key] = load_html_page(page, rescale_size)

    return pages


def render_bounding_boxes(image: Image, bounding_boxes: List[BoundingBox]) -> Image:
    """
    Render a set of bounding boxes (for human interpretability).
    """
    # start by creating the overlay that we're going to use to render the
    # bounding boxes
    poly = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(poly)

    for b in bounding_boxes:
        draw.rectangle(((b.x1, b.y1), (b.x2, b.y2)), fill=(128, 128, 128, 128))

    image.paste(poly, mask=poly)

    return image


def load_image(base: Path, pdf_page: PDFPage) -> Image:
    """Load an image of a pdf page."""
    for file in base.glob(f'{pdf_page.filename}*.png'):
        page_num = int(file.stem.split('-')[-1])
        if page_num == pdf_page.page_num:
            image = Image.open(str(file))
            return image
    return None


def load_image_from_filename(filename: Path) -> Image:
    """Load an image of a pdf page."""
    return Image.open(str(filename))


def pdf_page_from_image(image: Path) -> PDFPage:
    pdf_page = Path(image).stem
    pdf_page = pdf_page.split('-')
    pdf_page = PDFPage('-'.join(pdf_page[:-1]), int(pdf_page[-1]))
    return pdf_page


def load_annotations(csv: Path) -> Dict[PDFPage, List[BoundingBox]]:
    """
    Load the image annotations from MakeSense, exported as a CSV.
    """
    annotations = defaultdict(list)
    with csv.open() as fp:
        for line in fp:
            label, x, y, width, height, image, _, _ = line.strip().split(',')
            x, y, width, height = int(x), int(y), int(width), int(height)
            bounding_box = BoundingBox(**{
                'x1': x, 'y1': y,
                'x2': x+width, 'y2': y+height,
                'label': label
            })

            pdf_page = pdf_page_from_image(image)

            annotations[pdf_page].append(bounding_box)

    return annotations


def get_iow(word: BoundingBox, annotation: BoundingBox) -> float:
    """
    Calculate the Intersection over Word (IoW) of two bounding boxes.
    """
    # TODO: Move these assertions into the BoundingBox namedtuple init
    assert word.x1 < word.x2
    assert word.y1 < word.y2
    assert annotation.x1 < annotation.x2
    assert annotation.y1 < annotation.y2

    # determine the coordinates of the intersection rectangle
    x_left = max(word.x1, annotation.x1)
    y_top = max(word.y1, annotation.y1)
    x_right = min(word.x2, annotation.x2)
    y_bottom = min(word.y2, annotation.y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    word_area = (word.x2 - word.x1) * (word.y2 - word.y1)

    iow = intersection_area / word_area
    assert iow >= 0.0
    assert iow <= 1.0
    return iow


def merge(words: List[Tuple[BoundingBox, str]]) -> List[Tuple[List[BoundingBox], str]]:
    merged = []
    groups = itertools.groupby(words, lambda w: w[1])
    for group, contents in groups:
        merged.append(([w[0] for w in contents], group))
    return merged


def group_to_funsd(group_id: int, group: Tuple[List[BoundingBox], str]):
    box_x1 = min([w.x1 for w in group[0]])
    box_y1 = min([w.y1 for w in group[0]])
    box_x2 = max([w.x2 for w in group[0]])
    box_y2 = max([w.y2 for w in group[0]])
    return {
        "id": group_id,
        "text": " ".join([w.label for w in group[0]]),
        "box": [int(box_x1), int(box_y1), int(box_x2), int(box_y2)],
        "label": group[1],
        "words": [
            {
                "text": w.label,
                "box": [int(w.x1), int(w.y1), int(w.x2), int(w.y2)]
            } for w in group[0]
        ]
    }

def to_funsd_format(annotations: Dict[PDFPage, List[BoundingBox]],
                    pdfs: Dict[PDFPage, List[BoundingBox]]):
    pages = {}
    for page, boxes in annotations.items():
        if page not in pdfs:
            print(f'Missing: {page.filename}.pdf, page: {page.page_num} from pdfs')
            continue
        words = pdfs[page]

        prepared = []
        for word in words:
            word_label = 'other'
            for box in boxes:
                if get_iow(word, box) > 0.5:
                    word_label = box.label
            prepared.append((word, word_label))
        prepared = merge(prepared)
        form = {
            "form": [
                group_to_funsd(i, group) for i, group in enumerate(prepared)
            ]
        }

        pages[page] = form

    return pages


def to_blank_funsd(pdfs: Dict[PDFPage, List[BoundingBox]]):
    pages = {}
    for page, words in pdfs.items():
        prepared = [(word, 'other') for word in words]
        prepared = merge(prepared)
        form = {
            "form": [
                group_to_funsd(i, group) for i, group in enumerate(prepared)
            ]
        }

        pages[page] = form

    return pages
