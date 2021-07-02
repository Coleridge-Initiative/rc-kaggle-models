Hello!

Below you can find a outline of how to reproduce our solution for the Coleridge Initiative - Show US the Data competition.
If you run into any trouble with the setup/code or have any questions please contact us at nguyenquananhminh@gmail.com and tuankhoi94@gmail.com

### ARCHIVE CONTENTS

1.  notebooks/                              : Contains all notebooks to preprocess data for train/valid
2.  data/                                   : An original kaggle dataset, augmentation, extra labels, scispacy pretrained
3.  pretrained/                             : Pretrained bert models here
4.  processed_data/                         : A preprocessed data for training here
5.  callbacks.py                            : A callback layers for calculate JaccardF0.5 metric for valid data
6.  data_loader.py                          : Data sampling and augmentation here
7.  inference.py                            : A script for extract support embeddings
8.  directory_structure.txt                 : A directory structure of the project after finish all steps
9.  metric_layers.py                        : Include an implementation for ArcFace layer.
10. model.py                                : An implementation of main deep metric learning model
11. README.md                               : README for the code
12. requirements.txt                        : Needed python library for training model
13. setting.json                            : Contains all fixed path for raw data, processed data, pretrained and trained model
14. train.py                                : A script for training model
15. training.sh                             : A sh script for training all model


### HARDWARE: (The following specs were used to create the original solution)
Deep Learning AMI (Ubuntu 18.04) Version 46.0 AWS EC2

1. 8 Core CPU
2. 64 GB RAM
3. p3.2xlarge instance (1x V100 16GB VRAM)


### SOFTWARE (python packages are detailed separately in `requirements.txt`):

1. Python 3.7.10
2. CUDA tookit: 11.0
3. cuddn version: 8.
4. nvidia drivers: 450.119.03

### DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)

```bash
cd ./data
kaggle competitions download -c coleridgeinitiative-show-us-the-data
unzip coleridgeinitiative-show-us-the-data.zip
rm -rf ./coleridgeinitiative-show-us-the-data.zip
cd ..
```

### Step-By-Step To Reproduce Our Result

1. Run `pip install -r requirements.txt` to install needed libraries.
2. Run `notebooks/get_candidate_labels.ipynb` to get candidate labels by using scispacy and our custom algorithm.
3. Run `notebooks/preprocess.ipynb` to create training/validation data.
4. `python ./pretrained/setup_pretrained.py`. This script will download `scibert-base-case` and `biomed_roberta_base` from huggingface hub and save them into the `pretrained` folder.
5. `sh training.sh`. This script will train two models above and save the trained model into a `./saved_models/${MODEL_NAME}/run1`.
6. Pre Extract Support Embedding, MASK/NoMASK Embedding for inference:

    ```
    python inference.py extract-support-embedding \
        --model_name ${MODEL_NAME} \
        --pretrained_path ${YOUR_TRAINED_PATH} \
        --saved_path ${YOUR_SAVED_PATH} \
        --batch_size ${YOUR_BATCH_SIZE}
    ```
    For MODEL_NAME, we only support `biomed_roberta_base` and `scibert-base-cased`. Below is an example command line:
    ```
    python inference.py extract-support-embedding \
        --model_name biomed_roberta_base \
        --pretrained_path ./saved_models/biomed_roberta_base/run1/8-0.665-0.623-0.70--0.282.h5 \
        --saved_path ./saved_models/biomed_roberta_base/run1/ \
        --batch_size 128
    ```
6. Upload all files in saved_models/{MODEL_NAME} to Kaggle and run the inference though our shared submission kernel [kernel](https://www.kaggle.com/dathudeptrai/biomed-roberta-scibert-base). Note that the version we used for latest submission is version 5.


### Important Notes

Please make sure that you've read our MODEL SUMMARY documentation. Below are some important notes that we want to share to help you reproduce the results easily:

1.  As explained in MODEL SUMMARY, we only use 1/8 negative samples for training the model and did not set seed to make a better diversity for ensembling later. The result in leaderboard won't be affected too much by this factor (let say 0.x%). 
2. All models are trained with 10 epochs. Almost models are converged at the 9th or 10th epoch.
3. The valid score (JaccardF0.5) is just for reference since we didn't have a groundtruth so the exact value of valid score is not important. We observed that the jaccard-based Fbeta0.5 at 9th or 10th epoch is the best for all run. The model at 9th or 10th epoch is also got the best score on public leaderboard in our experiments.
4. The checkpoint is saved with the form A-B-C-D-E.h5 where:

    - A: is an epoch number. If a epoch is 9 then A is 8.
    - B: is a best valid JaccardF0.5 score at the NER threshold D
    - C: is an average valid JaccardF0.5 score at NER threshold in the list [0.5, 0.55, 0.6, 0.65, 0.7].
    - D: is a NER threshold at the best valid JaccardF0.5 score.
    - E: is a COSINE threshold at the best valid query classification F1 score.
5. You should train each model with multiple times (around 2-3 times) and take a best model based on a valid JaccardF0.5 score. You can do this by changing the `n_exps` in `training.sh` to 2 or 3.
6. As explained above, we didn't have a groundtruth of the validation data so all threshold that we found in the training process is just for reference. Please see section 2.5.1 of MODEL SUMMARY for the NER threshold we utilized. In the inference kernel, we used a lesser value for COSINE threshold; for example, if the optimum COSINE threshold for query classification F1 score is -0.282, we will use a value of -0.5.
7. You could get a private score around 0.58 with our solution. We also retrain and attach our trained model in `saved_models` directory after clean the code. The results of the retrain models are consistent with the latest submission models.
