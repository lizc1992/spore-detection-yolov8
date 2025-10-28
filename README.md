# Spores Counter Tool by LO

AI-based Method for Detecting and Counting Bacterial Spores in Microscopy Images.
This repository implements three main stages: **preprocessing**, **training**, and **inference**.

## Table of Contents

1. [Features](#features)
2. [Getting Started](#getting-started)
3. [Preprocessing](#preprocessing)
4. [Training](#training)
5. [Inference](#inference)
6. [Directory Structure](#directory-structure)

## Features

* Data ingestion & cleaning pipeline to convert raw input into model-ready form (preprocessing).
* Training script with configurable hyper-parameters, checkpointing, and logging.
* Inference module to load a trained model and apply to new data for predictions.

## Getting Started

  ```bash
  git clone https://github.com/lizc1992/Spore.git  
  cd Spore  
  ```
* Install dependencies:

  ```bash
  pip install -r requirements.txt  
  ```

## Preprocessing

This stage prepares raw data into the format needed for training the model.

### What it does

* Reads raw inputs (e.g., text, images, sensor logs) from `data/raw/`
* Cleans, filters, normalises, transforms features
* Splits data into train/validation/test sets
* Saves processed data into `data/processed/`

### Manual Tagging and Mask Correction

Manual tagging is performed using our in-house annotation tool.
Each image is tagged twice:
On the background_masked image – to mark the regions that should be added to the mask.
On the contour_masked image – to mark the regions that should be added along the object contours.

After tagging, the notebook `fix_labels_after_manual_tags.ipynb` is used to automatically re-generate updated masks incorporating these manual corrections.

## Training

This stage builds the model using the processed data.

### What it does

* Loads processed data from `data/processed/`
* Instantiates a model architecture 
* Defines loss, optimizer, training loop, and validation logic
* Saves best-performing checkpoint(s) to `models/`
* Logs training progress (loss, metrics) to console and to a log directory

### How to run

```bash
python training/train.py 
```

## Inference

This stage loads a trained model and uses it for predictions on new/unseen data.

### What it does

* Loads one or more model checkpoint(s) from `models/`
* Loads input data (raw)
* Applies any preprocessing steps required (mirroring training)
* Generates output predictions (e.g., class labels, probabilities, embeddings)
* Saves results to directory.

### How to run

```bash
python inference/run_inference.py 
```


## Directory Structure

```
Spore/
├── data/
│   ├── raw/
│   ├── processed/
│   └── new/                
├── config/
│   ├── preprocess.yaml
│   ├── train.yaml
│   └── inference.yaml
├── preprocessing/
│   └── run_preprocess.py
├── training/
│   └── train.py
├── inference/
│   └── run_inference.py
├── models/
│   └── best_model.pt
├── results/
│   └── predictions.csv
├── requirements.txt
└── README.md
```

