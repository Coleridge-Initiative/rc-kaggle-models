#!/Users/djpassadore/Python3/kaggle_env/bin/python3 -tt
# -*- coding: UTF-8 -*-

import pickle
import pandas as pd
import sys
import gensim
from gensim import corpora
from gensim import models
from gensim.models import Word2Vec
import multiprocessing
import spacy
import json

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

sentences = []
print('Loading raw sentences ...')
with open(RAW_DATA_DIR + 'raw_sentences_larger5_4_classification.bin', 'rb') as fp:   #Pickling
  sentences = pickle.load(fp)

nlp = spacy.load('en_core_web_sm')

# Calculating word frequencies
print('Calculating word frequencies in', len(sentences), 'sentences ... (Tick every 100,000 sentences)')
frequency = {}
n = 0
for sentence in sentences:
  doc = nlp(sentence)
  for tok in doc:
    if tok.text in frequency:
      frequency[tok.text] += 1
    else:
      frequency[tok.text] = 1
  n += 1
  if n%100000 == 0:
    sys.stdout.write(".")
    sys.stdout.flush()    
print('\nDone with', len(frequency), 'terms.')

print('Processing to elliminate low-frequency tokens ...')
processed_corpus = []
for sentence in sentences:
  ltokens = []
  doc = nlp(sentence)
  for tok in doc:
    # if not tok.is_punct and not tok.like_url and not tok.like_num and not tok.like_email:
    if frequency[tok.text] >= 7:
      ltokens.append(tok.text)
  processed_corpus.append(ltokens)

print('Generating dictionary ...')
dictionary = corpora.Dictionary(processed_corpus)
print('Saving dictionary ...')
dictionary.save_as_text(RAW_DATA_DIR + 'dict_w2v_freq7-spacy.txt')

np = multiprocessing.cpu_count()

print("Training word2vec model ...")

model = Word2Vec(sg = 0,          # Training algorithm: 1 for skip-gram; otherwise CBOW
                 hs = 1,          # If 1, hierarchical softmax will be used for model training. If 0, negative sampling will be used
                 min_count=7,     # Ignore words that appear less than this
                 vector_size=50,  # Dimensionality of word embeddings
                 workers=np,      # Number of processors (parallelisation)
                 window=11,        # Context window for words during training
                 epochs=1)        # Number of epochs training over corpus

print("Building vocabulary ...")
model.build_vocab(processed_corpus)

print('Training word2vec model ...')
for i in range(5):
  model.train(processed_corpus,
              compute_loss = True,
              total_examples=model.corpus_count,
              epochs = model.epochs)
  training_loss_val = model.get_latest_training_loss()
  print('After epoch %d: latest training loss is %f' %
            (i + 1, training_loss_val))

# print(model.vector_size)
# print(len(model.wv.key_to_index))

print("Saving word2vec model ...")
model.save(MODEL_CHECKPOINT_DIR + 'w2v-model-w11-f7-50-spacy.tmp')
word_vectors = model.wv
word_vectors.save_word2vec_format(MODEL_CHECKPOINT_DIR + 'gensim_vectors')
