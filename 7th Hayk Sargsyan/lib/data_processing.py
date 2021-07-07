import re
import spacy
import copy
import random
import logging
import json
from spacy.matcher import PhraseMatcher
from spacy.tokens import DocBin
from collections import defaultdict

from .utils import get_dataset_groups, get_candidate_group_mapping, is_group_unclear

log = logging.getLogger(__name__)


class DataProcessor():
    def __init__(self, df, candidate_groups):
        self.df = df
        self.groups = candidate_groups
        self.candidate_group_mapping = get_candidate_group_mapping(self.groups)
        self.additional_terms = []
        self.match_terms = ()
        self.val_groups = [
            'Rural-Urban Continuum Codes (RUCC)',
            'Agricultural Resource Management Survey (ARMS)'
        ]

    def _clean_label(self, txt):
        return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())

    def _is_intersecting(self, prev_spans, s, e):
        for span in prev_spans:
            span_s, span_e = span[0], span[1]
            span_range = set(range(span_s, span_e))
            new_span_range = set(range(s, e))
            if span_range.intersection(new_span_range):
                return True
        return False

    def _merge_spans(self, spans):
        new_spans = []
        start, end = spans[0]
        for span in spans:
            cur_start, cur_end = span
            if cur_start != start:
                if not self._is_intersecting(new_spans, start, end):
                    new_spans.append((start, end))
                start, end = cur_start, cur_end
            else:
                end = max(end, cur_end)
        if not self._is_intersecting(new_spans, start, end):
            new_spans.append((start, end))
        return new_spans

    def match_entities(self, add_terms=False):

        nlp = spacy.load('en_core_web_lg', disable=['ner', 'tok2vec'])
        self.candidate_group_mapping = get_candidate_group_mapping(self.groups)
        terms = set(self.candidate_group_mapping.keys())
        terms = {term for term in terms if len(term) > 2}
        if add_terms:
            terms.update(self.additional_terms)
            terms.discard('bDMARDs')
            terms.discard('WW3')
        long_terms = {term.lower() for term in terms
                      if len(term) > 6 and len(term.split()) > 2}
        long_terms = sorted(list(long_terms), reverse=True, key=lambda x: len(x))
        short_terms = {term for term in terms
                       if len(term) <= 6 or len(term.split()) <= 2}
        short_terms = sorted(list(short_terms), reverse=True, key=lambda x: len(x))
        self.match_terms = (short_terms, long_terms)
        long_patterns = list(nlp.tokenizer.pipe(long_terms))
        short_patterns = list(nlp.tokenizer.pipe(short_terms))
        matcher_long = PhraseMatcher(nlp.vocab, attr='LOWER')
        matcher_long.add(f'TerminologyListLong', long_patterns)
        matcher_short = PhraseMatcher(nlp.vocab)
        matcher_short.add(f'TerminologyListShort', short_patterns)

        data = defaultdict(list)
        id_sentences = self.df.groupby('Id')['sentences'].agg(list)
        for Id in self.df['Id'].unique():
            sentences = [sent for sent in id_sentences[Id][0]
                         if 'data' in sent.lower()]
            for doc in nlp.pipe(sentences):
                matches_long = matcher_long(doc)
                if matches_long:
                    merged_spans = self._merge_spans(
                        [
                            (start, end)
                            for _, start, end in matches_long
                        ]
                    )
                    data[Id].append(
                        (doc.text,
                         {
                             'entities': [],
                             'spans': []
                         })
                    )
                    for start, end in merged_spans:
                        data[Id][-1][1]['spans'].append((start, end))
                        data[Id][-1][1]['entities'].append(
                            (doc[start].idx, doc[end-1].idx+len(doc[end-1]))
                        )
                    data[Id][-1] = tuple(data[Id][-1])
                    matches_short = matcher_short(doc)
                    if matches_short:
                        merged_spans = self._merge_spans(
                            [(start, end) for _, start, end in matches_short]
                        )

                        for start, end in merged_spans:
                            if not self._is_intersecting(data[Id][-1][1]['spans'], start, end):
                                data[Id][-1][1]['spans'].append((start, end))
                                data[Id][-1][1]['entities'].append(
                                    (doc[start].idx, doc[end-1].idx+len(doc[end-1]))
                                )
                        data[Id][-1] = tuple(data[Id][-1])
                else:
                    matches_short = matcher_short(doc)
                    if matches_short:
                        merged_spans = self._merge_spans(
                            [(start, end) for _, start, end in matches_short])
                        data[Id].append(
                            (doc.text,
                             {
                                 'entities': [],
                                 'spans': []
                             })
                        )
                        for start, end in merged_spans:
                            data[Id][-1][1]['spans'].append((start, end))
                            data[Id][-1][1]['entities'].append(
                                (doc[start].idx, doc[end-1].idx+len(doc[end-1]))
                            )
                        data[Id][-1] = tuple(data[Id][-1])

        return data

    def merge_candidates(self):
        matched_data = self.match_entities()
        for items in matched_data.values():
            for item in items:
                sentence = item[0]
                ents = sorted(item[1]['entities'], key=lambda x: x[0])
                for i in range(len(ents) - 1):
                    if ents[i][1] == ents[i+1][0] - 1:
                        ent1 = sentence[ents[i][0]:ents[i][1]]
                        ent2 = sentence[ents[i+1][0]:ents[i+1][1]]
                        if ent1 == ent1.upper() and ent2 == ent2.upper():
                            continue
                        clean_ent2 = self._clean_label(ent2)
                        if (clean_ent2 in self.candidate_group_mapping and
                                len(self.candidate_group_mapping[clean_ent2]) == 1):
                            group = self.candidate_group_mapping[clean_ent2][0]
                            ents[i+1] = (ents[i][0], ents[i+1][1])
                            new_candidate = sentence[ents[i+1][0]:ents[i+1][1]]
                            if new_candidate not in self.groups[group]:
                                self.groups[group].append(new_candidate)


class TrainDataProcessor(DataProcessor):
    def __init__(self, df, candidate_groups):
        super().__init__(df, candidate_groups)
        self.additional_terms = [
            'BMI = body mass index',
            'and cardiovascular disease',
            'and mild cognitive impairment',
            'MCI = mild cognitive impairment',
            'and Mild Cognitive Impairment',
            'T1-weighted spoiled gradient echo',
            'T1-weighted Spoiled Gradient Recalled',
            'sum-of-boxes Clinical Dementia Rating',
            'T2-weighted fluid attenuated inversion recovery',
            'T1 weighted magnetization prepared rapid acquisition gradient echo',
            'T1-weighted magnetization prepared rapid gradient echo',
            'and Statistical Parametric Mapping',
            'UDS neuropsychological test',
            'T2-weighted magnetic resonance imaging',
            'and statistical parametric mapping',
            'DNA Next Generation Sequencing',
            'FSL Brain Extraction Tool',
            'OECD Programme for International Student Assessment',
            'ENSO sea surface temperature',
            'OECD Program for International Student Assessment',
            'SSTAs',
            'CATI = computer-assisted telephone interview',
            'National Hurricane Center (NHC) Best Track',
            'NOAA vertical datum transformation tool',
            'S&T = science and technology',
            'FEMA flood insurance studies',
            'FSA farm loan programs',
            'SARS-CoV-2 receptor-binding domain'
        ]

    def construct_data(self):
        dataset_groups = get_dataset_groups(self.df, self.groups)
        self.merge_candidates()
        matched_data = self.match_entities(add_terms=True)

        not_allowed_groups = [k for k in self.groups
                              if is_group_unclear(k[:k.rfind('(')].strip())]
        not_allowed_nodata_cands = set()
        for k in not_allowed_groups:
            not_allowed_nodata_cands.update([cand.lower() for cand in self.groups[k]])
        add_nodata_cands = [
            'oecd programme for international student assessment',
            'oecd program for international student assessment',
            's&t = science and technology'
        ]
        not_allowed_nodata_cands.update(add_nodata_cands)

        allowed_nodata_cands = set()
        short_terms, long_terms = self.match_terms
        for term in long_terms:
            if term not in not_allowed_nodata_cands:
                allowed_nodata_cands.add(term)

        for term in short_terms:
            if term.lower() not in not_allowed_nodata_cands:
                allowed_nodata_cands.add(term)
        allowed_nodata_cands.add('general circulation models')

        id_labels = self.df.groupby('Id')['dataset_label'].agg(set)
        data_labels = {}
        for Id in self.df['Id'].unique():
            labels = copy.deepcopy(id_labels[Id])
            for label in labels.copy():
                if label in dataset_groups:
                    for title in dataset_groups[label]:
                        labels.update([cand for cand in self.groups[title]
                                       if cand.lower().startswith(label.lower())])
            data_labels[Id] = labels
        data_labels_lowered = {k: set([cand.lower() for cand in v])
                               for k, v in data_labels.items()}

        full_train_data = []
        random.seed(0)
        for Id, items in matched_data.items():
            for item in items:
                item_data = []
                sentence = item[0]
                for ent in item[1]['entities']:
                    candidate = sentence[ent[0]:ent[1]]
                    if candidate.lower() in data_labels_lowered[Id]:
                        label = 'DATA'
                    elif candidate.lower() in allowed_nodata_cands:
                        label = 'NODATA'
                    else:
                        continue
                    masked_sentence = sentence[:ent[0]] + \
                        '@CAND@' + sentence[ent[1]:]
                    item_data.append(
                        (masked_sentence,
                         {
                             'Id': Id, 'candidate': candidate, 'label': label
                         })
                    )
                if any([ex[1]['label'] == 'DATA' for ex in item_data]):
                    full_train_data.extend(item_data)
                elif random.uniform(0, 1) < 0.25:
                    full_train_data.extend(item_data)
        return full_train_data

    def split_data(self, data):
        all_ids = {item[1]['Id'] for item in data}
        val_labels = set()
        for group in self.val_groups:
            val_labels.update(self.groups[group])
        val_ids = set()
        for example in data:
            if example[1]['candidate'] in val_labels:
                val_ids.add(example[1]['Id'])
        train_ids = {Id for Id in all_ids if Id not in val_ids}
        train_data = [item for item in data if item[1]['Id'] in train_ids]
        val_data = [item for item in data if item[1]['Id'] in val_ids]
        log.info(f'Number of training sentences {len(train_data)}')
        log.info(f'Number of validation sentences {len(val_data)}')
        train_positive = len([item for item in train_data
                              if item[1]['label'] == 'DATA'])
        valid_positive = len([item for item in val_data
                              if item[1]['label'] == 'DATA'])
        train_negative = len([item for item in train_data
                              if item[1]['label'] == 'NODATA'])
        valid_negative = len([item for item in val_data
                              if item[1]['label'] == 'NODATA'])
        log.info(f'Number of positive examples in the training set {train_positive}')
        log.info(f'Number of positive examples in the validation set {valid_positive}')
        log.info(f'Number of negative examples in the training set {train_negative}')
        log.info(f'Number of negative examples in the validation set {valid_negative}')

        return (train_data, val_data)

    def _make_docs(self, data):

        docs = []
        nlp = spacy.load('en_core_web_lg', disable=['ner'])
        for doc, anns in nlp.pipe(data, as_tuples=True):
            label = anns['label']
            if label == 'DATA':
                doc.cats['positive'] = 1
                doc.cats['negative'] = 0
            else:
                doc.cats['positive'] = 0
                doc.cats['negative'] = 1

            # put them into a nice list
            docs.append(doc)

        return docs

    def convert_to_spacy_format(self, train, valid):
        random.seed(0)
        random.shuffle(train)
        random.shuffle(valid)

        train_docs = self._make_docs(train)
        train_bin = DocBin(docs=train_docs)
        valid_docs = self._make_docs(valid)
        valid_bin = DocBin(docs=valid_docs)
        return train_bin, valid_bin

    def save_data(self, data, f_path):
        data.to_disk(f_path)


class TestDataProcessor(DataProcessor):
    def __init__(self, df, candidate_groups):
        super().__init__(df, candidate_groups)

    def match_and_merge_entities(self):

        nlp = spacy.load('en_core_web_lg', disable=['ner', 'tok2vec'])
        self.candidate_group_mapping = get_candidate_group_mapping(self.groups)
        terms = set(self.candidate_group_mapping.keys())
        terms = {term for term in terms if len(term) > 2}
        long_terms = {term.lower() for term in terms
                      if len(term) > 6 and len(term.split()) > 2}
        long_terms = sorted(list(long_terms), reverse=True, key=lambda x: len(x))
        short_terms = {term for term in terms
                       if len(term) <= 6 or len(term.split()) <= 2}
        short_terms = sorted(list(short_terms), reverse=True, key=lambda x: len(x))
        self.match_terms = (short_terms, long_terms)
        long_patterns = list(nlp.tokenizer.pipe(long_terms))
        short_patterns = list(nlp.tokenizer.pipe(short_terms))
        matcher_long = PhraseMatcher(nlp.vocab, attr='LOWER')
        matcher_long.add(f'TerminologyListLong', long_patterns)
        matcher_short = PhraseMatcher(nlp.vocab)
        matcher_short.add(f'TerminologyListShort', short_patterns)

        data = defaultdict(list)
        id_sentences = self.df.groupby('Id')['sentences'].agg(list)
        for Id in self.df['Id'].unique():
            sentences = [sent for sent in id_sentences[Id][0]
                         if 'data' in sent.lower()]
            for doc in nlp.pipe(sentences):
                matches_long = matcher_long(doc)
                if matches_long:
                    merged_spans = self._merge_spans(
                        [
                            (start, end)
                            for _, start, end in matches_long
                        ]
                    )
                    data[Id].append(
                        (doc.text,
                         {
                             'entities': [],
                             'spans': []
                         })
                    )
                    for start, end in merged_spans:
                        data[Id][-1][1]['spans'].append((start, end))
                        data[Id][-1][1]['entities'].append(
                            (doc[start].idx, doc[end-1].idx+len(doc[end-1]))
                        )
                    data[Id][-1] = tuple(data[Id][-1])
                    matches_short = matcher_short(doc)
                    if matches_short:
                        merged_spans = self._merge_spans(
                            [(start, end) for _, start, end in matches_short]
                        )

                        for start, end in merged_spans:
                            if not self._is_intersecting(data[Id][-1][1]['spans'], start, end):
                                data[Id][-1][1]['spans'].append((start, end))
                                data[Id][-1][1]['entities'].append(
                                    (doc[start].idx, doc[end-1].idx+len(doc[end-1]))
                                )
                        data[Id][-1] = tuple(data[Id][-1])
                else:
                    matches_short = matcher_short(doc)
                    if matches_short:
                        merged_spans = self._merge_spans(
                            [(start, end) for _, start, end in matches_short])
                        data[Id].append(
                            (doc.text,
                             {
                                 'entities': [],
                                 'spans': []
                             })
                        )
                        for start, end in merged_spans:
                            data[Id][-1][1]['spans'].append((start, end))
                            data[Id][-1][1]['entities'].append(
                                (doc[start].idx, doc[end-1].idx+len(doc[end-1]))
                            )
                        data[Id][-1] = tuple(data[Id][-1])

        for items in data.values():
            for j, item in enumerate(items):
                indices_to_remove = set()
                sentence = item[0]
                item[1]['entities'] = sorted(item[1]['entities'],
                                             key=lambda x: x[0])
                for i in range(len(item[1]['entities']) - 1):
                    if item[1]['entities'][i][1] == item[1]['entities'][i+1][0] - 1:
                        ent1 = sentence[item[1]['entities'][i][0]:item[1]['entities'][i][1]]
                        ent2 = sentence[item[1]['entities'][i+1][0]:item[1]['entities'][i+1][1]]
                        if ent1 == ent1.upper() and ent2 == ent2.upper():
                            continue
                        clean_ent2 = self._clean_label(ent2)
                        if (clean_ent2 in self.candidate_group_mapping and
                                len(self.candidate_group_mapping[clean_ent2]) == 1):
                            group = self.candidate_group_mapping[clean_ent2][0]
                            item[1]['entities'][i+1] = (item[1]['entities'][i][0],
                                                        item[1]['entities'][i+1][1])
                            new_candidate = sentence[item[1]['entities']
                                                     [i+1][0]:item[1]['entities'][i+1][1]]
                            if new_candidate not in self.groups[group]:
                                self.groups[group].append(new_candidate)
                            indices_to_remove.add(i)
                if indices_to_remove:
                    items[j][1]['entities'] = [ex for i, ex in enumerate(
                        items[j][1]['entities']) if i not in indices_to_remove]

        return data

    def construct_data(self):
        matched_data = self.match_and_merge_entities()
        data = []
        for Id, items in matched_data.items():
            for item in items:
                sentence = item[0]
                for ent in item[1]['entities']:
                    candidate = sentence[ent[0]:ent[1]]
                    masked_sentence = sentence[:ent[0]] + '@CAND@' + sentence[ent[1]:]
                    data.append(
                        (masked_sentence,
                         {
                             'Id': Id, 'candidate': candidate
                         })
                    )
        return data

    def naive_search(self):
        nlp = spacy.load('en_core_web_lg', disable=['ner'])
        terms = set([f'{cand} dataset' for cand in self.candidate_group_mapping.keys()])

        terms = sorted(list(terms), reverse=True, key=lambda x: len(x))
        patterns = list(nlp.tokenizer.pipe(terms))
        matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        matcher.add(f'TerminologyListDataset', patterns)

        data = defaultdict(list)
        id_sentences = self.df.groupby('Id')['sentences'].agg(list)
        for Id in self.df['Id'].unique():
            sentences = [sent for sent in id_sentences[Id][0]
                         if 'data' in sent.lower()]
            for doc in nlp.pipe(sentences):
                matches = matcher(doc)
                if matches:
                    merged_spans = self._merge_spans(
                        [(start, end) for _, start, end in matches]
                    )
                    data[Id].append(
                        (doc.text,
                         {
                             'entities': [],
                             'spans': []
                         })
                    )
                    for start, end in merged_spans:
                        data[Id][-1][1]['spans'].append((start, end))
                        data[Id][-1][1]['entities'].append(
                            (doc[start].idx, doc[end-1].idx+len(doc[end-1])))
                    data[Id][-1] = tuple(data[Id][-1])
        return data

    def save_data(self, data, f_path):
        with open(f_path, 'w') as f:
            json.dump(data, f, indent=2)
