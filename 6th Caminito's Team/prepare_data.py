#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#Upload Kaggle Data
train_df = pd.read_csv('../../coleridgeinitiative-show-us-the-data/train.csv')
sample_sub = pd.read_csv('../../coleridgeinitiative-show-us-the-data/sample_submission.csv')
train_files_path = '../../coleridgeinitiative-show-us-the-data/train'
test_files_path = '../../coleridgeinitiative-show-us-the-data/test'


# In[16]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


tqdm.pandas()
train_df['text_all'] = train_df['Id'].progress_apply(partial(read_append_return, output='all'))


# In[9]:


tqdm.pandas()
train_df['text_all'] = train_df['text_all'].progress_apply(text_cleaning)


# In[21]:


to_vec = pd.read_csv(RAW_DATA_DIR + 'cleanedLabel_toVec_Diego_for_vectors.csv')


# In[13]:


#Prepare Dataset list
temp_1 = [text_cleaning(x) for x in train_df['dataset_label'].unique()]
temp_2 = [text_cleaning(x) for x in train_df['dataset_title'].unique()]
temp_3 = [text_cleaning(x) for x in train_df['cleaned_label'].unique()]
temp_4 = [text_cleaning(x) for x in to_vec['0'].unique()]

existing_labels = set(temp_1 + temp_2 + temp_3 + temp_4)

print(len(existing_labels))


# In[15]:


import spacy
from scispacy.abbreviation import AbbreviationDetector
nlp = spacy.load("en_core_sci_sm", disable = ['ner', 'tagger', 'attribute_ruler', 'lemmatizer', 'tok2vec'])
nlp.max_length = 10000000
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
print(nlp.pipe_names)


# In[16]:


#Prepare spaCy entity-ruler(For sentences that have the word dataset or a synonym.)
syn_dataset = ['dataset','datasets', 'data-set', 'data-sets', 'data sets', 'data set', 'datum', 'databases', 'database', 'data bank', 'data banks', 'databank', 'databanks', 'metadata', 'raw data', 'time series', 'time-series']
patterns = []

for dataset in syn_dataset:
    phrase = []
    for word in nlp(dataset):
        pattern = {}
        pattern["LOWER"] = str(word)
        phrase.append(pattern)
    #patterns.append({"label": dataset, "pattern": phrase})
    patterns.append({"label": "DATASET", "pattern": phrase})

len(patterns)


# In[17]:


from spacy.pipeline import EntityRuler
ruler = EntityRuler(nlp, overwrite_ents=True)
nlp.add_pipe(ruler)
ruler.add_patterns(patterns)
print(nlp.pipe_names)


# In[21]:


#Drop duplicates papers
train_df_unique = train_df.drop_duplicates(subset=['Id'])
print (len(train_df_unique['Id']))


# In[17]:


#Upload organizations list
int_org_df = pd.read_csv(RAW_DATA_DIR + 'International_ORG_sin_duplicados.csv')
list3 = int_org_df['Text_Org'].tolist()


# In[19]:


#List of dataset
list2 = existing_labels
list2_to_l = list(list2)
len(list2_to_l)


# In[22]:


#Obtengo datos de Dataset syn ruler.
sent = []
start = []
end = []
label = []
label_text = []
label_long_text = []
data_train = []

for doc in tqdm(nlp.pipe(train_df_unique['text_all'][:5], batch_size=50)):
    abrv_text = []
    for ent in doc.ents:
        for abrv in doc._.abbreviations:
            text = abrv._.long_form.text
            if text not in list3 and not abrv.text.islower() and not text.islower():                
                #print(abrv.sent, text)
                if abrv.sent == ent.sent:
                    print(abrv._.long_form.text + " (" + abrv.text + ")")
                    #print(ent.sent)
                    #print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")                    
                    number = abrv.start_char - abrv._.long_form.end_char
                    if 0 <= number <= 5:
                        #print(abrv.sent)
                        #print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
                        #print(abrv.start_char, abrv._.long_form.end_char)
                        start_ = abrv._.long_form.start_char - abrv.sent.start_char
                        end_ = (abrv.end_char + 1) - abrv.sent.start_char
                        #print(abrv.sent.start_char, start_, end_, abrv.end_char - abrv.sent.start_char, abrv.sent.end_char)
                        sentence = abrv.sent
                        ent_text = abrv._.long_form.text + " (" + abrv.text + ")"
                        #print(ent_text)
                        sent.append(sentence.text)
                        start.append(start_)
                        end.append(end_)
                        label.append("DATASET")
                        label_text.append(ent_text)
                        label_long_text.append(abrv._.long_form.text)
                        data_train.append("SI")
                    else:  
                        start_ = abrv.start_char - abrv.sent.start_char
                        end_ = abrv.end_char - abrv.sent.start_char
                        sentence = abrv.sent
                        ent_text = abrv.text
                        sent.append(sentence.text)
                        start.append(start_)
                        end.append(end_)
                        label.append("DATASET")
                        label_text.append(ent_text)
                        label_long_text.append(abrv._.long_form.text)
                        data_train.append("SI")
                else:
                    number = abrv.start_char - abrv._.long_form.end_char
                    if 0 <= number <= 5:
                        #print(abrv.sent)
                        #print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
                        #print(abrv.start_char, abrv._.long_form.end_char)
                        start_ = abrv._.long_form.start_char - abrv.sent.start_char
                        end_ = (abrv.end_char + 1) - abrv.sent.start_char
                        #print(abrv.sent.start_char, start_, end_, abrv.end_char - abrv.sent.start_char, abrv.sent.end_char)
                        sentence = abrv.sent
                        ent_text = abrv._.long_form.text + " (" + abrv.text + ")"
                        print(ent_text)
                        sent.append(sentence.text)
                        start.append(start_)
                        end.append(end_)
                        label.append("DATASET")
                        label_text.append(ent_text)
                        label_long_text.append(abrv._.long_form.text)
                        data_train.append("NO")
                    else:  
                        start_ = abrv.start_char - abrv.sent.start_char
                        end_ = abrv.end_char - abrv.sent.start_char
                        sentence = abrv.sent
                        ent_text = abrv.text
                        sent.append(sentence.text)
                        start.append(start_)
                        end.append(end_)
                        label.append("DATASET")
                        label_text.append(ent_text)
                        label_long_text.append(abrv._.long_form.text)
                        data_train.append("NO")


# In[23]:


sentence_df = pd.DataFrame()
sentence_df['sentence'] = sent
sentence_df['start'] = start
sentence_df['end'] = end
sentence_df['label'] = label
sentence_df['label_text'] = label_text
sentence_df['label_long_text'] = label_long_text
sentence_df['data_train'] = data_train
sentence_df['ent_freq'] = sentence_df.groupby('label_text')['label_text'].transform('count')
sentence_df['sent_freq'] = sentence_df.groupby('sentence')['sentence'].transform('count')
sentence_df


# In[24]:


sentence_df.to_csv(RAW_DATA_DIR + 'dataset_example_to_do_the_analysis_manually.csv')


# ## Remember that one of the most complex processes was the manual cleaning of the sentences, which is the next step.
