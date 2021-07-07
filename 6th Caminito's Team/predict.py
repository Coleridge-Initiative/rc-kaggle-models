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


# In[3]:


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


# In[4]:


train_df.head(3)


# In[5]:


def text_cleaning(text):
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
        #print("FEDE")
        return all_data


# In[7]:


#tqdm.pandas()
#train_df['text_all'] = train_df['Id'].progress_apply(partial(read_append_return, output='all'))


# In[8]:


#tqdm.pandas()
#train_df['text_all'] = train_df['text_all'].progress_apply(text_cleaning)


# In[9]:


tqdm.pandas()
sample_sub['text_all'] = sample_sub['Id'].progress_apply(partial(read_append_return, train_files_path=test_files_path, output='all'))


# In[10]:


tqdm.pandas()
sample_sub['text_all'] = sample_sub['text_all'].progress_apply(text_cleaning)


# In[11]:


sample_sub.head()


# In[12]:


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


# In[13]:


import spacy
from spacy.language import Language
from typing import Tuple, List, Optional, Set, Dict
from collections import defaultdict
from spacy.tokens import Span, Doc
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler


# In[14]:


def find_abbreviation(
    long_form_candidate: Span, short_form_candidate: Span
) -> Tuple[Span, Optional[Span]]:
    """
    Implements the abbreviation detection algorithm in "A simple algorithm
    for identifying abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    The algorithm works by enumerating the characters in the short form of the abbreviation,
    checking that they can be matched against characters in a candidate text for the long form
    in order, as well as requiring that the first letter of the abbreviated form matches the
    _beginning_ letter of a word.

    Parameters
    ----------
    long_form_candidate: Span, required.
        The spaCy span for the long form candidate of the definition.
    short_form_candidate: Span, required.
        The spaCy span for the abbreviation candidate.

    Returns
    -------
    A Tuple[Span, Optional[Span]], representing the short form abbreviation and the
    span corresponding to the long form expansion, or None if a match is not found.
    """
    long_form = " ".join([x.text for x in long_form_candidate])
    short_form = " ".join([x.text for x in short_form_candidate])

    long_index = len(long_form) - 1
    short_index = len(short_form) - 1

    while short_index >= 0:
        current_char = short_form[short_index].lower()
        # We don't check non alpha-numeric characters.
        if not current_char.isalnum():
            short_index -= 1
            continue

            # Does the character match at this position? ...
        while (
            (long_index >= 0 and long_form[long_index].lower() != current_char)
            or
            # .... or if we are checking the first character of the abbreviation, we enforce
            # to be the _starting_ character of a span.
            (
                short_index == 0
                and long_index > 0
                and long_form[long_index - 1].isalnum()
            )
        ):
            long_index -= 1

        if long_index < 0:
            return short_form_candidate, None

        long_index -= 1
        short_index -= 1

    # The last subtraction will either take us on to a whitespace character, or
    # off the front of the string (i.e. long_index == -1). Either way, we want to add
    # one to get back to the start character of the long form
    long_index += 1

    # Now we know the character index of the start of the character span,
    # here we just translate that to the first token beginning after that
    # value, so we can return a spaCy span instead.
    word_lengths = 0
    starting_index = None
    for i, word in enumerate(long_form_candidate):
        # need to add 1 for the space characters
        word_lengths += len(word.text_with_ws)
        if word_lengths > long_index:
            starting_index = i
            break

    return short_form_candidate, long_form_candidate[starting_index:]

def filter_matches(
    matcher_output: List[Tuple[int, int, int]], doc: Doc
) -> List[Tuple[Span, Span]]:
    # Filter into two cases:
    # 1. <Short Form> ( <Long Form> )
    # 2. <Long Form> (<Short Form>) [this case is most common].
    candidates = []
    for match in matcher_output:
        start = match[1]
        end = match[2]
        # Ignore spans with more than 8 words in them, and spans at the start of the doc
        if end - start > 8 or start == 1:
            continue
        if end - start > 3:
            # Long form is inside the parens.
            # Take one word before.
            short_form_candidate = doc[start - 2 : start - 1]
            long_form_candidate = doc[start:end]
        else:
            # Normal case.
            # Short form is inside the parens.
            short_form_candidate = doc[start:end]

            # Sum character lengths of contents of parens.
            abbreviation_length = sum([len(x) for x in short_form_candidate])
            max_words = min(abbreviation_length + 5, abbreviation_length * 2)
            # Look up to max_words backwards
            long_form_candidate = doc[max(start - max_words - 1, 0) : start - 1]

        # add candidate to candidates if candidates pass filters
        if short_form_filter(short_form_candidate):
            candidates.append((long_form_candidate, short_form_candidate))

    return candidates

def short_form_filter(span: Span) -> bool:
    # All words are between length 2 and 10
    if not all([2 <= len(x) < 10 for x in span]):
        return False

    # At least 50% of the short form should be alpha
    if (sum([c.isalpha() for c in span.text]) / len(span.text)) < 0.5:
        return False

    # The first character of the short form should be alpha
    if not span.text[0].isalpha():
        return False
    return True

class AbbreviationDetector:
    """
    Detects abbreviations using the algorithm in "A simple algorithm for identifying
    abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    This class sets the `._.abbreviations` attribute on spaCy Doc.

    The abbreviations attribute is a `List[Span]` where each Span has the `Span._.long_form`
    attribute set to the long form definition of the abbreviation.

    Note that this class does not replace the spans, or merge them.
    """

    def __init__(self, nlp) -> None:
        Doc.set_extension("abbreviations", default=[], force=True)
        Span.set_extension("long_form", default=None, force=True)

        self.matcher = Matcher(nlp.vocab)
        self.matcher.add(
            "parenthesis", None, [{"ORTH": "("}, {"OP": "+"}, {"ORTH": ")"}]
        )
        self.global_matcher = Matcher(nlp.vocab)

    def find(self, span: Span, doc: Doc) -> Tuple[Span, Set[Span]]:
        """
        Functional version of calling the matcher for a single span.
        This method is helpful if you already have an abbreviation which
        you want to find a definition for.
        """
        dummy_matches = [(-1, int(span.start), int(span.end))]
        filtered = filter_matches(dummy_matches, doc)
        abbreviations = self.find_matches_for(filtered, doc)

        if not abbreviations:
            return span, set()
        else:
            return abbreviations[0]
    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        matches_no_brackets = [(x[0], x[1] + 1, x[2] - 1) for x in matches]
        filtered = filter_matches(matches_no_brackets, doc)
        occurences = self.find_matches_for(filtered, doc)

        for (long_form, short_forms) in occurences:
            for short in short_forms:
                short._.long_form = long_form
                doc._.abbreviations.append(short)
        return doc

    def find_matches_for(
        self, filtered: List[Tuple[Span, Span]], doc: Doc
    ) -> List[Tuple[Span, Set[Span]]]:
        rules = {}
        all_occurences: Dict[Span, Set[Span]] = defaultdict(set)
        already_seen_long: Set[str] = set()
        already_seen_short: Set[str] = set()
        for (long_candidate, short_candidate) in filtered:
            short, long = find_abbreviation(long_candidate, short_candidate)
            # We need the long and short form definitions to be unique, because we need
            # to store them so we can look them up later. This is a bit of a
            # pathalogical case also, as it would mean an abbreviation had been
            # defined twice in a document. There's not much we can do about this,
            # but at least the case which is discarded will be picked up below by
            # the global matcher. So it's likely that things will work out ok most of the time.
            new_long = long.string not in already_seen_long if long else False
            new_short = short.string not in already_seen_short
            if long is not None and new_long and new_short:
                already_seen_long.add(long.string)
                already_seen_short.add(short.string)
                all_occurences[long].add(short)
                rules[long.string] = long
                # Add a rule to a matcher to find exactly this substring.
                self.global_matcher.add(
                    long.string, None, [{"ORTH": x.text} for x in short]
                )
        to_remove = set()
        global_matches = self.global_matcher(doc)
        for match, start, end in global_matches:
            string_key = self.global_matcher.vocab.strings[match]
            to_remove.add(string_key)
            all_occurences[rules[string_key]].add(doc[start:end])
        for key in to_remove:
            # Clean up the global matcher.
            self.global_matcher.remove(key)
        return list((k, v) for k, v in all_occurences.items())


# In[15]:


#This model is a previous NER-LAST-VEC-1100-RULER model, but has the same entity-ruler and vectors.
#That is why we use it for string matching and vectors similarity.

nlp = spacy.load(MODEL_CHECKPOINT_DIR + "NER-LAST-VEC-716-RULER", disable = ['tagger', 'ner'])


# In[16]:


# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
Language.factories['AbbreviationDetector'] = lambda nlp, **cfg: AbbreviationDetector(nlp, **cfg)
nlp.max_length = 15000000
print(nlp.pipe_names)


# In[17]:


int_org_df = pd.read_csv(RAW_DATA_DIR + 'International_ORG_sin_duplicados.csv')
list3 = int_org_df['Text_Org'].tolist()

to_vec = pd.read_csv(RAW_DATA_DIR + 'cleanedLabel_toVec_Diego_for_vectors.csv')
list2 = [text_cleaning(x).lower().rstrip() for x in to_vec['0'].unique()]

list2 = set(list2)
len(list2)


# In[18]:


nlp2 = spacy.load(MODEL_CHECKPOINT_DIR + 'Ensemble_30000_textcat')
nlp3 = spacy.load(MODEL_CHECKPOINT_DIR + 'NER-LAST-VEC-1100-RULER', disable = ['tagger', 'parser'])
nlp4 = spacy.load(MODEL_CHECKPOINT_DIR + 'Ensemble_3000_dataset_vs_ORG')

print(nlp4.pipe_names)


# In[19]:


#Vectors
list2SpacyDocs = [nlp(x) for x in list2]


# In[20]:


#train_df_unique = train_df.drop_duplicates(subset=['Id'])
#print (len(train_df_unique['Id']))


# In[21]:


#Abvr-NER-Sentences Combino 2 modelos nlp
id_list = []
lables_list = []
threshold = 0.90
threshold_data = 0.95
ne = 0
maxlen=1000000

for index, row in tqdm(sample_sub.iterrows()):
    llabels = []
    papertxt = row['text_all']
    row_id = row['Id']
    nlp_labels = []
    docs = []
    if len(papertxt) > maxlen:
        i = round(len(papertxt)/maxlen)-1
        while i >= 0:
            docs.append(papertxt[0:maxlen-1])
            papertxt = papertxt[maxlen:]
            i = i - 1
    else:
        docs.append(papertxt)
        
    abvr_list = []
    if docs:
        for text in docs:
            doc = nlp(text)
            for abrv in doc._.abbreviations:
                if any(abrv.text in s for s in abvr_list):
                    pass
                else:
                    if abrv._.long_form.text not in list3 and abrv.text != "AD" and not abrv.text.islower() and not abrv._.long_form.text.islower(): 
                        #print(abrv.text)
                        long_data = abrv._.long_form.text + ", " + abrv.text
                        if abvr_list:
                            if any(abrv.text in s for s in abvr_list):
                                pass
                            else:
                                abvr_list.append(long_data)                           
                        else:
                            abvr_list.append(long_data)
                            
            #print(abvr_list)
            en = 0
            for ent in doc.ents:
                en += 1
                if ent.label_ == "RULDATA":                    
                    #print("In list: ", ent.text)
                    if any(ent.text.lower() in s.lower() for s in abvr_list):
                        for words in abvr_list:
                            palabras = words.split(", ")
                            #print(palabras[0])
                            abrv_word = palabras[-1]
                            #print(abrv_word)
                            if abrv_word in ent.text or palabras[0] in ent.text:
                                #print(ent.text)
                                nlp_labels.append(words)
                                nlp_labels.append(abrv_word)                             
                    else:                        
                        if len(ent) > 2:
                            long_ent = ent.text
                            long_ent_find = re.findall('\(([^)]+)', long_ent)
                            if long_ent_find:
                                abrv_ent = re.search('\(([^)]+)', long_ent).group(1)                        
                                nlp_labels.append(long_ent)
                                nlp_labels.append(abrv_ent)
                            else:
                                nlp_labels.append(long_ent)
            #print(abvr_list)
            #Vectores
            labels_vec_set = set(nlp_labels)
            #print(labels_vec_set)
            label_vec = [nlp(x.lower()) for x in labels_vec_set]
            np = 0
            for sent in doc.sents:
                slen = len(sent.text.split())
                if slen >= 7 and slen <= 100:
                    strsent = str(sent.text)
                    strsent = strsent.strip()
                    doccat = nlp2(strsent)
                    if doccat.cats['SI'] > threshold:
                        np += 1
                        docent = nlp3(strsent)
                        for ent in docent.ents:                            
                            if any(ent.text.lower() in s.lower() for s in abvr_list) and not any(ent.text.lower() in s.lower() for s in labels_vec_set):
                                #print("In list: ", ent.text)
                                for s in abvr_list:                                    
                                    if ent.text.lower() in s.lower():                                            
                                        palabras = s.split(", ")
                                        data_vs_org = nlp4(palabras[0].lower())                                
                                        #print(ent.text)
                                        if data_vs_org.cats['SI'] > threshold_data:                                            
                                            nlp_labels.append(palabras[0])
                                            nlp_labels.append(palabras[-1]) 
                                            
                            else:
                                paso = False
                                text1 = nlp(ent.text.lower())                                
                                data_vs_org = nlp4(ent.text)                                
                                #print(ent.text)
                                if data_vs_org.cats['SI'] > threshold_data:
                                    paso = True                                    
                                    for vec in label_vec:
                                        #print("Lista ruler: ", vec)
                                        similarity2 = text1.similarity(vec)
                                        #print(similarity2)
                                        #print("Lista ner: ", text1)
                                        #print("........")                                        
                                        if similarity2 > 0.90:
                                            #print(ent.text)
                                            paso = False
                                if paso:
                                    #print(ent.text)
                                    nlp_labels.append(ent.text)                
            
    cleaned_labels = [clean_text(x) for x in nlp_labels]
    #cleaned_labels =[]
    cleaned_labels = set(cleaned_labels)
    lables_list.append('|'.join(cleaned_labels))
    id_list.append(str(row_id))
    #print("LABELS")
    print(cleaned_labels)
    print('Sentences classified as positive:', np)
    print('Rul-Ents classified as positive:', en)


# In[22]:


submission = pd.DataFrame()
submission['Id'] = id_list
submission['PredictionString'] = lables_list


# In[26]:


submission.to_csv(SUBMISSION_DIR + 'submission.csv', index=False)
get_ipython().system("head 'submissions/submission.csv'")


# In[27]:


#!find . -type d > directory_structure.txt


# In[28]:


#!pip freeze > requirements.txt

