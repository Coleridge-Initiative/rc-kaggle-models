# Entry Points

## Usage
1. For training run the following command in the terminal `sh train.sh`
- This will run the data preparation script for the training set:
```bash
python prepare_data.py --train
```
    - Read training data from RAW_DATA_DIR (specified in SETTINGS.json)
    - Run any preprocessing steps and generate candidates
    - Save the spacy binaries in TRAIN_DATA_CLEAN_DIR (specified in SETTINGS.json)
    - Takes about 45 minutes
- and then run the training:
```bash
python -m spacy train ./train_configs/config.cfg \
    --output ./models \
    --paths.train ./data/processed/traiin/train.spacy \
    --paths.dev ./data/processed/train/valid.spacy \
    --gpu-id 0 > logs/train.log
```
    - Read in training and validation sets from TRAIN_DATA_CLEAN_DIR
    - Train the model
        - Will donload the transformer model if not available in the system
    - Save the model in MODEL_CHECKPOINT_DIR (specified in SETTINGS.json)
    - Takes around 30 minutes

2. For prediction run the following command in the terminal `sh predict.sh`
- This will run the data preparation script for the training set:
```bash
python prepare_data.py --test
```
    - Read test data from RAW_DATA_DIR (specified in SETTINGS.json)
    - Run any preprocessing steps and generate candidates
    - Save the necessary files for the inference in TEST_DATA_CLEAN_DIR (specified in SETTINGS.json)
    - Takes around 45 minutes
- and then run the predict script:
```bash
python predict.py
```
    - Read in the necessary files from TEST_DATA_CLEAN_DIR
    - Predict and postprocess the predictions
    - Save the predictions in SUBMISSION_DIR (specified in SETTINGS.json)
    - Takes around 45 minutes
