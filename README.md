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

1. split_images.py - split each high-resolution microscopy image (2160*3840) into four smaller patches (1080*1920)
2. Manual Tagging and Mask Correction
   Manual tagging is performed using our in-house annotation tool (`tools/spore_marking_tool.py`).
   Each image is tagged twice:
     On the background_masked image – to mark the regions that should be added to the mask.
     On the contour_masked image – to mark the regions that should be added along the object contours.
   After tagging, the notebook `fix_labels_after_manual_tags.ipynb` is used to automatically re-generate updated masks incorporating these manual corrections.
3. convert_json_to_yolo8seg_format.py - This script converts manually tagged annotation files (typically in JSON format) into the YOLOv8 segmentation format, which is required for model training.
4. apply_augmentation.py - Applied extensive data augmentation using the `Albumentations` library.


## Training

This stage builds the model using the processed data.

### What it does

* Loads processed data
* Instantiates a model architecture 
* Defines loss, optimizer, training loop, and validation logic
* Saves best-performing checkpoint(s) to `models/`
* Logs training progress (loss, metrics) to console and to a log directory


## Inference

This stage loads a trained model and uses it for predictions on new/unseen data.

### What it does

* Loads one or more model checkpoint(s) from `models/`
* Loads input data (raw)
* Applies any preprocessing steps required (mirroring training)
* Generates output predictions (e.g., class labels, probabilities, embeddings)
* Saves results to directory.


## Directory Structure

```
Spore/
├── data/
│   ├── raw/
│   ├── processed/               
├── config/
│   ├── cells_aug.yaml
│   └── evaluation.yaml
├── preprocessing/
│   ├── split_images.py
│   ├── fix_labels_after_manual_tags.ipynb
│   ├── convert_json_to_yolo8seg_format.py
│   └── apply_augmentation.py
├── training/
│   └── train.py
├── inference/
│   └── evaluation.py
├── requirements.txt
└── README.md
```

