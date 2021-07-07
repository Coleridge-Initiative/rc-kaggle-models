#!/bin/bash

echo "> Preparinng training data..."
python prepare_data.py --train

echo "> Training classification model..."
python -m spacy train ./train_configs/config.cfg \
    --output ./models \
    --paths.train ./data/processed/train/train.spacy \
    --paths.dev ./data/processed/train/valid.spacy \
    --gpu-id 0 > logs/train.log
