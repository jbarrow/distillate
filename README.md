# 🧪`distillate`

🧪`distillate` is a Python package to aid with information extraction from PDFs.
It allows you to annotate PDF text using bounding boxes, and train multi-modal models.

## Installation

### Requirements

For 🧪`distillate` to work, you must first install `Poppler`, a unix PDF annotation tool.
In future versions, I'd like to get OS X and Windows installation working.

```
sudo apt update
sudo apt install poppler
```

### Pip

Once the above requirements are met, you can install via:

```
pip install -e .
```

## Usage

There are 6 commands that make up the `distillate` core.
The first step is to create a new dataset from a collection of PDFs (and optionally annotations):

```
distillate create [DATASET] --pdfs=[PDFs] --base=[BASE]
```

If you add more PDFs or annotations, you can use `distillate update` to add them to the dataset:

```
distillate update [DATASET] --pdfs=[PDFs] --annotations=[ANNOTATIONS]
```

And if you would like to visualize annotations for the dataset (or for a specific file), you can use `distillate visualize`:

```
distillate visualize [DATASET] --filename=[FILE] --base=[BASE][DATASET]
```

To train a LayoutLM model, use `distillate train` with a specific dataset:

```
distillate train [DATASET] --model=[MODEL] --split=[SPLIT_PERCENTAGE]
```

If you'd like to use a pretrained model instead, you can use `distillate download` to fetch one of the available models:

```
distillate download [MODEL]
```

Using a pretrained model, you can run over a PDF using `distillate predict`, passing an optional serialization directory to store the intermediate outputs:

```
distillate predict [FILE] --model=[MODEL] --inspect/--no-inspect --serialization-dir=[SERIALIZATION_DIRECTORY]
```

## Annotation Format

We use the CSV annotations from MakeSense.ai.

## Tutorial
