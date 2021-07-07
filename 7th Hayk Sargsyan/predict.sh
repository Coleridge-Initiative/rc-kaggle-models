#!/bin/bash

echo "> Preparinng test data..."
python prepare_data.py --test

echo "> Running prediction..."
python predict.py
