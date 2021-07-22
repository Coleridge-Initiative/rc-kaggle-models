# Coleridge Initiative - Show US the Data: Final model (Hayk Sargsyan)

Hello!

Below is a outline of how to reproduce the solution for the Coleridge Initiative - Show US the Data competition.
The files and codes are provided by [Hayk Sargsyan](https://www.kaggle.com/sarhayk).

## ARCHIVE CONTENTS
- `models/model-best`: contains the model binaries used in generating the solution
- `lib`: Contains all the code necessary for reproducing the solution
- `prepare_data.py`: the python script that preprocesses the data
- `predict.py`: the python script that generates predictions
- `train.sh`: bash script for training
- `predict.sh`: bash script for generating the predictions
- `train_predict.sh`: bash script to retrain and predict
- `requirements.txt`: contains the versions of packages used in obtaining the original results

## HARDWARE: (The following specs were used to create the original solution)
- Ubuntu 20.04 LTS (512 GB boot disk)
- 16 vCPUs, 124 GB memory
- 1 x NVIDIA GTX 1080 TI

## SOFTWARE (python packages are detailed separately in `requirements.txt`):
- Python 3.7.10
- CUDA 10.1
- cuddn 8.0.5
- nvidia drivers 460.73.01

## DATA SETUP
assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api is installed.

```bash
$ cd data
$ kaggle competitions download -c coleridgeinitiative-show-us-the-data
$ unzip coleridgeinitiative-show-us-the-data.zip
$ rm -f coleridgeinitiative-show-us-the-data.zip
$ cd ..
```

## MODEL BUILD: There are two options to produce the solution.

All directories are specified in `settings.json`

1. Only prediction (assumes the trained model is in MODEL_CHECKPOINT_DIR)
    - expect this to run for 1.5 hrs
    - preprocesses the raw test data located in RAW_DATA_DIR
    - saves/overwrites the files needed for the inference in the TEST_DATA_CLEAN_DIR directory
    - runs the inference (overwrites `submission.csv` file in SUBMISSION_DIR)

2. Retrain models and predicting
    - expect this to run for about 3 hrs
    - preprocesses the raw training data located in RAW_DATA_DIR
    - saves/overwrites the spacy binary files needed for the training in the TRAIN_DATA_CLEAN_DIR directory
    - runs the training (overwrites the binary model in MODEL_CHECKPOINT_DIR)
    - Runs the prediction detailed in option 1


### shell commands to run each build is below
The details on the python script calls are given in `entry_points.md`

1. Only prediction
```bash
$ sh predict.sh
```

2. Retrain and predict
```bash
$ sh train_predict.sh
```

The training can also be run separately, without the prediction step, via
```bash
$ sh train.sh
```
- expect this to run for 1.5 hrs
