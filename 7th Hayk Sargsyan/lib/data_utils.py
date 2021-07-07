import re
import string
import json
import os
import logging
import pandas as pd
from functools import partial
from spacy.pipeline import Sentencizer
from spacy.lang.en import English

log = logging.getLogger(__name__)

# Load basic English model and sentencizer
nlp = English()
sentencizer = Sentencizer()


def clean_label(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())


def read_texts(filename, pub_dir=None, output='all'):
    """Reads in the publication text and appends to the dataframe

    Credit: https://www.kaggle.com/prashansdixit/coleridge-initiative-eda-baseline-model
    """
    json_path = os.path.join(pub_dir, f'{filename}.json')
    headings = []
    contents = []
    combined = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            headings.append(data.get('section_title'))
            contents.append(data.get('text'))
            combined.append(data.get('section_title'))
            combined.append(data.get('text'))

    all_headings = ' '.join(headings)
    all_contents = ' '.join(contents)
    all_data = '. '.join(combined)

    if output == 'text':
        return all_contents
    elif output == 'head':
        return all_headings
    else:
        return all_data


def sanitize_text(text):

    # Remove quotes
    text = text.replace('"', '')

    allowed_chars = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = re.sub(r'\s', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = ''.join(
        [k for k in text if k in allowed_chars or k in string.punctuation]
    )

    # Remove repeated sequences (char and spaces)
    repeated_seq = re.compile(r"([\w\.]\s*)\1{5,}")
    matches = [match.group(0) for match in repeated_seq.finditer(text)]
    matches = sorted(matches, reverse=True, key=lambda x: len(x))
    for match in matches:
        text = re.sub(re.escape(match), '', text)

    text = re.sub(' +', ' ', text)
    matches = [match.group(0) for match in repeated_seq.finditer(text)]
    matches = sorted(matches, reverse=True, key=lambda x: len(x))
    for match in matches:
        text = re.sub(re.escape(match), '', text)

    return text


def clean_text(text):
    return re.sub('[^A-Za-z0-9]+', ' ', str(text).lower()).strip()


def extract_sentences(text):
    """Extracts the sentences and return all < 1500 lenghth sentences"""

    # Remove the "." after "al." to have a cleaner sentence extraction
    text = re.sub(r" al.", " al", text)

    # Make sure works properly on long texts
    nlp.max_length = max(1000000, len(text) + 1)

    doc = nlp(text)
    return [sent.text.strip() for sent in sentencizer(doc).sents
            if len(sent.text) < 1500]


def load_data(df_path, pub_dir=None):
    """Reads in the dataframe and the publication texts"""

    # Read in the data apply some sanitization
    df = pd.read_csv(df_path)
    if pub_dir is None:
        return df

    log.info('Reading the publication texts...')
    df['text'] = df['Id'].apply(partial(read_texts, pub_dir=pub_dir))

    log.info('Sanitizing the texts...')
    df['text'] = df['text'].apply(sanitize_text)

    return df
