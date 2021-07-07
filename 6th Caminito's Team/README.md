## Hardware used to train spaCy NLP models.
***
* System:
    Host: fede-dsk Kernel: 5.4.0-77-generic x86_64 bits: 64 
    Desktop: Cinnamon 4.8.6 Distro: Linux Mint 20.1 Ulyssa 
* Machine:
    Mobo: ASUSTeK model: PRIME Z490-A v: Rev 1.xx
* CPU:
    8-Core: Intel Core i7-10700K type: MT MCP speed: 800 MHz 
    min/max: 800/5100 MHz 
* Graphics:
    Device-1: NVIDIA TU104 [GeForce RTX 2080 SUPER] driver: nvidia v: 460.80  
* Drives:
    Local Storage: total: 2.29 TiB used: 46.90 GiB (2.0%) 
* Info:
    Processes: 352 Uptime: 43m Memory: 31.26 GiB used: 2.89 GiB (9.3%) 
    Shell: bash inxi: 3.0.38


## Hardware used to train Gensim Vectors (python packages are detailed separately in `mac_gensim_requirements.txt`):
***
* Model Name:              MacBook Pro
* Model Identifier         MacBookPro14,2
* Processor Name:          Dual-Core Intel Core i5
* Processor Speed:         3.1 GHz
* Number of Processors:    1
* Total Number of Cores:   2
* L2 Cache (per Core):     256 KB
* L3 Cache:                4 MB
* Hyper-Threading Technology: Enabled
* Memory:                  8 GB
* System Firmware Version: 429.120.4.0.0
* SMC Version (system):    2.4416

## SOFTWARE (python packages are detailed separately in `requirements.txt`):
***
1. Python 3.8.5 
2. spaCy v2.3.5 (https://v2.spacy.io/)
3. scispaCy (https://github.com/allenai/scispacy)
4. Gensim (https://github.com/RaRe-Technologies/gensim)
5. Jupyter Notebook and Conda 4.10.1 were used.


## MODEL BUILD: There are three options to produce the solution.
1) very fast prediction
    a) runs in a few minutes
    b) uses precomputed neural network predictions

### shell command to run each build is below
1) very fast prediction (Model for training NER)
!python -m spacy train en models/NER-LAST-VEC-1100 'data/processed/train_last_sent_929_format.json' 'data/processed/dev_last_sent_164_format.json' --base-model 'en_core_web_sm' --vectors 'models/gensim_vectors'  -p ner -R

## To train TEXTCAT models
***
* Use train Jupyter Notebook or train.py

## To make new predictions
***
* Use predict Jupyter Notebook or predict.py

## To train Gensim Vectors
***
* Use generate_vectors_with_spacy_v2.py and mac_gensim_requirements.txt
