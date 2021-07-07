#!/usr/bin/env python
# coding: utf-8

# # Training NER

# In[4]:


get_ipython().system("python -m spacy train en models/NER-LAST-VEC-1100 'data/processed/train_last_sent_929_format.json' 'data/processed/dev_last_sent_164_format.json' --base-model 'en_core_web_sm' --vectors 'models/gensim_vectors'  -p ner -R")


# ### Adding Entity Ruler

# In[7]:


# Asthetics https://www.kaggle.com/manabendrarout/tabular-data-preparation-basic-eda-and-baseline
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Basic
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import json
import os
import random
from tqdm.autonotebook import tqdm
import string
import re
from functools import partial
from ipywidgets import IntProgress


# In[4]:


#Upload Kaggle Data
train_df = pd.read_csv('../../coleridgeinitiative-show-us-the-data/train.csv')
sample_sub = pd.read_csv('../../coleridgeinitiative-show-us-the-data/sample_submission.csv')
train_files_path = '../../coleridgeinitiative-show-us-the-data/train'
test_files_path = '../../coleridgeinitiative-show-us-the-data/test'


# In[8]:


# Opening JSON file
f = open('SETTINGS.json',)
  
# returns JSON object as 
# a dictionary
data = json.load(f)

RAW_DATA_DIR = data['RAW_DATA_DIR']
TRAIN_DATA_CLEAN_PATH = data['TRAIN_DATA_CLEAN_PATH']
TEST_DATA_CLEAN_PATH = data['TEST_DATA_CLEAN_PATH']
MODEL_CHECKPOINT_DIR = data['MODEL_CHECKPOINT_DIR']
SUBMISSION_DIR = data['SUBMISSION_DIR']
f.close()


# In[5]:


def text_cleaning(text):
    '''
    Removes special charecters, multiple spaces
    text - Sentence that needs to be cleaned
    '''
    text = re.sub(r'[^A-Za-z0-9.!?'"'"'()\[\]]+', ' ', text)
    text = re.sub("'", '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('[.]{2,}', '.', text)
    text = re.sub(r'\. \.', '.', text)
    text = re.sub(r' \.', '.', text)    
    
    return text


# In[6]:


def read_append_return(filename, train_files_path=train_files_path, output='text'):
    json_path = os.path.join(train_files_path, (filename+'.json'))
    headings = []
    contents = []
    combined = []
    with open(json_path, 'r') as f: #encoding='utf-8'
        json_decode = json.load(f)
        for data in json_decode:
            headings.append(data.get('section_title'))
            s = text_cleaning(data.get('text'))
            if len(s) > 200000:
                #s = data.get('text')
                #print(data.get('text'))
                l = s.split()
                n = 100000
                texto = [" ".join(l[x:x+n]) for x in range(0, len(l), n)]   
                contents.extend(texto)
            else:
                contents.append(s)
                #print(contents1)
            combined.append(data.get('section_title'))
            combined.append(data.get('text'))
    
    all_headings = ' '.join(headings)
    all_contents = ' '.join(contents)
    #contents = contents1.extend(contents2)
    #print(combined)
    all_data = '. '.join(combined)
    
    if output == 'text':
        return all_contents
    elif output == 'head':
        return all_headings
    elif output == 'comb':
        return contents
    else:
        return all_data     


# In[7]:


tqdm.pandas()
train_df['text_all'] = train_df['Id'].progress_apply(partial(read_append_return, output='all'))


# In[8]:


tqdm.pandas()
train_df['text_all'] = train_df['text_all'].progress_apply(text_cleaning)


# In[9]:


to_vec = pd.read_csv(RAW_DATA_DIR + 'cleanedLabel_toVec_Diego_for_vectors.csv')


# In[10]:


#Sin minusculas
temp_1 = [text_cleaning(x).lower().rstrip() for x in train_df['dataset_label'].unique()]
temp_2 = [text_cleaning(x).lower().rstrip() for x in train_df['dataset_title'].unique()]
temp_3 = [text_cleaning(x).lower().rstrip() for x in train_df['cleaned_label'].unique()]
temp_4 = [text_cleaning(x).lower().rstrip() for x in to_vec['0'].unique()]
#temp_4 = [text_cleaning(x).lower().rstrip() for x in to_vec['title'].unique()]


existing_labels = set(temp_1 + temp_2 + temp_3 + temp_4)

#Minusculas y si espacio al final, para LOWER patterns
temp_1_low = [text_cleaning(x).rstrip() for x in train_df['dataset_label'].unique()]
temp_2_low = [text_cleaning(x).rstrip() for x in train_df['dataset_title'].unique()]
temp_3_low = [text_cleaning(x).rstrip() for x in to_vec['0'].unique()]

existing_labels_text = set(temp_1_low + temp_2_low + temp_3_low)

print(len(existing_labels), len(existing_labels_text))


# In[10]:


import spacy

nlp = spacy.load(MODEL_CHECKPOINT_DIR + 'NER-LAST-VEC-1100/model-best')


# In[12]:


#genero patterns para spacy Lower y text para len 1
patterns = []


for dataset in existing_labels:
    len_data = dataset.split()
    if len(len_data) > 2:
        #print(dataset)
        phrase = []
        for word in nlp(dataset):
            pattern = {}
            pattern["LOWER"] = str(word)
            phrase.append(pattern)
        #patterns.append({"label": dataset, "pattern": phrase})
        patterns.append({"label": "RULDATA", "pattern": phrase})


for dataset in existing_labels_text:
    len_data = dataset.split()
    if len(len_data) < 3:
        #print(dataset)
        phrase = []
        for word in nlp(dataset):
            pattern = {}
            pattern["TEXT"] = str(word)
            phrase.append(pattern)
        #patterns.append({"label": dataset, "pattern": phrase})
        patterns.append({"label": "RULDATA", "pattern": phrase})


len(patterns)


# In[13]:


from spacy.pipeline import EntityRuler
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler, after='ner')
ruler.add_patterns(patterns)
print(nlp.pipe_names)


# In[ ]:


#Saving NLP model
#nlp.to_disk(MODEL_CHECKPOINT_DIR +"NER-LAST-VEC-1100-RULER")


# # Training TEXTCAT Dataset recognizer

# In[11]:


# this code is modified from spaCy's user guide for TextCategorizer training 
from __future__ import unicode_literals, print_function
from __future__ import unicode_literals
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from sklearn.metrics import f1_score, accuracy_score, classification_report

import warnings 
warnings.simplefilter('ignore')

import re
#Lo saco de otro codigo para limpiar un poco
def clean_string(mystring):
    return re.sub('[^A-Za-z\ 0-9 ]+', '', mystring)

def main(model=None, n_iter=5, init_tok2vec=None):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("es")  # create blank Language class
        print("Created blank 'es' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat",
            config={
                "exclusive_classes": True,
                "architecture": "ensemble",
            }
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    for i in ['SI','NO']:
        textcat.add_label(i)
    
    # load the datasets
    print("Loading data...")
    #Train
    df = pd.read_csv(TRAIN_DATA_CLEAN_PATH + '3037_for_textcat_train.csv')
    #df.drop(['text'], axis=1, inplace=True)
    df = df[df['TextCat'] != 'empty']

    conclusion_values = df['TextCat'].unique()
    labels_default = dict((v, 0) for v in conclusion_values)

    train_data = []
    for i, row in df.iterrows():

        label_values = copy.deepcopy(labels_default)
        label_values[row['TextCat']] = 1

        train_data.append((str(row['sentence']), {"cats": label_values}))

    train_data = train_data[:5000]    

    
    #dev
    df_dev = pd.read_csv(TEST_DATA_CLEAN_PATH + '759_for_textcat_dev.csv')
    #df_dev.drop(['text'], axis=1, inplace=True)
    df_dev = df_dev[df_dev['TextCat'] != 'empty']

    conclusion_dev_values = df_dev['TextCat'].unique()
    labels_dev_default = dict((v, 0) for v in conclusion_dev_values)

    dev_data = []
    for i, row in df_dev.iterrows():

        label_dev_values = copy.deepcopy(labels_dev_default)
        label_dev_values[row['TextCat']] = 1

        dev_data.append((str(row['sentence']), {"cats": label_dev_values}))

    dev_data = dev_data
    
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            len(df['sentence']) + len(df_dev['sentence']), len(df['sentence']), len(df_dev['sentence'])
        )
    )
   
    #test
    df_test = pd.read_csv(TEST_DATA_CLEAN_PATH + '759_for_textcat_dev.csv')
    #df_test.drop(['text'], axis=1, inplace=True)
    df_test = df_test[df_test['TextCat'] != 'empty']

    conclusion_test_values = df_test['TextCat'].unique()
    labels_test_default = dict((v, 0) for v in conclusion_test_values)

    test_data = []
    for i, row in df_test.iterrows():

        label_test_values = copy.deepcopy(labels_test_default)
        label_test_values[row['TextCat']] = 1

        test_data.append((str(clean_string(row['sentence'])), {"cats": label_test_values}))

    test_data = test_data
    
    #Aca hago el cambio
    train_ls = train_data
    valid_ls = dev_data
    test_ls = test_data
    
    # Convert valid text and label to list.
    valid_text, valid_label = list(zip(*valid_ls))

    # Convert test text and label to list.
    test_text, test_label = list(zip(*test_ls))
    
    n_iter = 20
    print_every= 1
    not_improve = 5 


# Train model
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training() # initiate a new model with random weights
        if init_tok2vec is not None:
            init_tok2vec = Path(init_tok2vec)
            print("Loaded Vector model '%s'" % init_tok2vec)
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())                
        print("Training the model...")

        score_f1_best = 0
        early_stop = 0

        for i in range(n_iter):
            losses = {}
            true_labels = list() # true label
            pdt_labels = list() # predict label

            # batch up the examples using spaCy's minibatch
            random.shuffle(train_ls)  # shuffle training data every iteration
            batches = minibatch(train_ls, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)

            with textcat.model.use_params(optimizer.averages): 
                # evaluate on valid_text, valid_label
                docs = [nlp.tokenizer(text) for text in valid_text]

                for j, doc in enumerate(textcat.pipe(docs)):
                    true_series = pd.Series(valid_label[j]['cats'])
                    true_label = true_series.idxmax()  # idxmax() is the new version of argmax() 
                    true_labels.append(true_label)

                    pdt_series = pd.Series(doc.cats)
                    pdt_label = pdt_series.idxmax()  # idxmax() is the new version of argmax() 
                    pdt_labels.append(pdt_label)

                score_f1 = f1_score(true_labels, pdt_labels, average='weighted')
                score_ac = accuracy_score(true_labels, pdt_labels)

                if i % print_every == 0:
                    print('textcat loss: {:.4f}\tf1 score: {:.3f}\taccuracy: {:.3f}'.format(
                        losses['textcat'],score_f1, score_ac))

                if score_f1 > score_f1_best:
                    early_stop = 0
                    score_f1_best = score_f1
                    with nlp.use_params(optimizer.averages):
                        output_dir = Path(MODEL_CHECKPOINT_DIR + 'Ensemble_3000_dataset_vs_ORG')
                        if not output_dir.exists():
                              output_dir.mkdir()
                        nlp.to_disk(output_dir) # save the model
                else:
                    early_stop += 1

                if early_stop >= not_improve:
                    print('Finished training...')
                    break

                if i == n_iter:
                    print('Finished training...')
                #return {"textcat_a": score_ac, "textcat_l": losses['textcat'], "textcat_f": score_f1}


# In[17]:


#train_ensemble_model 
bb = main(model=None, init_tok2vec=None)


# # Training TEXTCAT Sentence recognizer

# In[ ]:


# this code is modified from spaCy's user guide for TextCategorizer training 
from __future__ import unicode_literals, print_function
from __future__ import unicode_literals
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from sklearn.metrics import f1_score, accuracy_score, classification_report

import warnings 
warnings.simplefilter('ignore')

import re
#Lo saco de otro codigo para limpiar un poco
def clean_string(mystring):
    return re.sub('[^A-Za-z\ 0-9 ]+', '', mystring)

def main(model=None, n_iter=5, init_tok2vec=None):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("es")  # create blank Language class
        print("Created blank 'es' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat",
            config={
                "exclusive_classes": True,
                "architecture": "ensemble",
            }
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    for i in ['SI','NO']:
        textcat.add_label(i)
    
    # load the datasets
    print("Loading data...")
    #Train
    df = pd.read_csv(TRAIN_DATA_CLEAN_PATH + '30000_for_textcat_train.csv')
    #df.drop(['text'], axis=1, inplace=True)
    df = df[df['TextCat'] != 'empty']

    conclusion_values = df['TextCat'].unique()
    labels_default = dict((v, 0) for v in conclusion_values)

    train_data = []
    for i, row in df.iterrows():

        label_values = copy.deepcopy(labels_default)
        label_values[row['TextCat']] = 1

        train_data.append((str(row['sentence']), {"cats": label_values}))

    train_data = train_data[:5000]    

    
    #dev
    df_dev = pd.read_csv(TEST_DATA_CLEAN_PATH + '30000_for_textcat_dev.csv')
    #df_dev.drop(['text'], axis=1, inplace=True)
    df_dev = df_dev[df_dev['TextCat'] != 'empty']

    conclusion_dev_values = df_dev['TextCat'].unique()
    labels_dev_default = dict((v, 0) for v in conclusion_dev_values)

    dev_data = []
    for i, row in df_dev.iterrows():

        label_dev_values = copy.deepcopy(labels_dev_default)
        label_dev_values[row['TextCat']] = 1

        dev_data.append((str(row['sentence']), {"cats": label_dev_values}))

    dev_data = dev_data
    
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            len(df['sentence']) + len(df_dev['sentence']), len(df['sentence']), len(df_dev['sentence'])
        )
    )
   
    #test
    df_test = pd.read_csv(TEST_DATA_CLEAN_PATH + '30000_for_textcat_dev.csv')
    #df_test.drop(['text'], axis=1, inplace=True)
    df_test = df_test[df_test['TextCat'] != 'empty']

    conclusion_test_values = df_test['TextCat'].unique()
    labels_test_default = dict((v, 0) for v in conclusion_test_values)

    test_data = []
    for i, row in df_test.iterrows():

        label_test_values = copy.deepcopy(labels_test_default)
        label_test_values[row['TextCat']] = 1

        test_data.append((str(clean_string(row['sentence'])), {"cats": label_test_values}))

    test_data = test_data
    
    #Aca hago el cambio
    train_ls = train_data
    valid_ls = dev_data
    test_ls = test_data
    
    # Convert valid text and label to list.
    valid_text, valid_label = list(zip(*valid_ls))

    # Convert test text and label to list.
    test_text, test_label = list(zip(*test_ls))
    
    n_iter = 20
    print_every= 1
    not_improve = 5 


# Train model
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training() # initiate a new model with random weights
        if init_tok2vec is not None:
            init_tok2vec = Path(init_tok2vec)
            print("Loaded Vector model '%s'" % init_tok2vec)
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())                
        print("Training the model...")

        score_f1_best = 0
        early_stop = 0

        for i in range(n_iter):
            losses = {}
            true_labels = list() # true label
            pdt_labels = list() # predict label

            # batch up the examples using spaCy's minibatch
            random.shuffle(train_ls)  # shuffle training data every iteration
            batches = minibatch(train_ls, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)

            with textcat.model.use_params(optimizer.averages): 
                # evaluate on valid_text, valid_label
                docs = [nlp.tokenizer(text) for text in valid_text]

                for j, doc in enumerate(textcat.pipe(docs)):
                    true_series = pd.Series(valid_label[j]['cats'])
                    true_label = true_series.idxmax()  # idxmax() is the new version of argmax() 
                    true_labels.append(true_label)

                    pdt_series = pd.Series(doc.cats)
                    pdt_label = pdt_series.idxmax()  # idxmax() is the new version of argmax() 
                    pdt_labels.append(pdt_label)

                score_f1 = f1_score(true_labels, pdt_labels, average='weighted')
                score_ac = accuracy_score(true_labels, pdt_labels)

                if i % print_every == 0:
                    print('textcat loss: {:.4f}\tf1 score: {:.3f}\taccuracy: {:.3f}'.format(
                        losses['textcat'],score_f1, score_ac))

                if score_f1 > score_f1_best:
                    early_stop = 0
                    score_f1_best = score_f1
                    with nlp.use_params(optimizer.averages):
                        output_dir = Path(MODEL_CHECKPOINT_DIR + 'Ensemble_30000_textcat')
                        if not output_dir.exists():
                              output_dir.mkdir()
                        nlp.to_disk(output_dir) # save the model
                else:
                    early_stop += 1

                if early_stop >= not_improve:
                    print('Finished training...')
                    break

                if i == n_iter:
                    print('Finished training...')
                #return {"textcat_a": score_ac, "textcat_l": losses['textcat'], "textcat_f": score_f1}


# In[ ]:


#train_ensemble_model 
bb = main(model=None, init_tok2vec=None)

