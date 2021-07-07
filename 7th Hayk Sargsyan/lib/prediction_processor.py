import re
import numpy as np
import logging
from collections import defaultdict, Counter
from .data_utils import clean_label
from .utils import is_org_like, is_data_like

log = logging.getLogger(__name__)


class Dataset():

    def __init__(self, groups):
        self.groups = groups
        self.label_group_mapping = self._label_group_mapping()
        self.scores = []
        self.data_groups = {}
        self.label_data_mapping = {}
        self.rejected = set()

    def _label_group_mapping(self):
        label_group_mapping = defaultdict(list)
        for k, v in self.groups.items():
            for label in v:
                cleaned_label = clean_label(label)
                if k not in label_group_mapping[cleaned_label]:
                    label_group_mapping[cleaned_label].append(k)
        return label_group_mapping

    def label_count(self, group):
        return len(self.label_group_mapping[group])

    def add_label(self, group, label, Id, score):
        if group not in self.data_groups:
            self.data_groups[group] = {
                'Ids': set(),
                'Id_scores': defaultdict(list),
                'labels': {}
            }
        self.data_groups[group]['Ids'].add(Id)
        self.data_groups[group]['Id_scores'][Id].append(score)
        if label not in self.data_groups[group]['labels']:
            self.data_groups[group]['labels'][label] = {
                'Ids': [],
                'scores': []
            }
        self.data_groups[group]['labels'][label]['Ids'].append(Id)
        self.data_groups[group]['labels'][label]['scores'].append(score)
        if label not in self.label_data_mapping:
            self.label_data_mapping[label] = []
        if group not in self.label_data_mapping[label]:
            self.label_data_mapping[label].append(group)

    def get_is_dataset_freq(self, group, threshold=0.5):
        n_dataset = len([Id for Id, scores in self.data_groups[group]['Id_scores'].items()
                         if np.max(scores) > threshold])
        return n_dataset / len(self.data_groups[group]['Id_scores'])

    def mean_group_score(self, group):
        scores = []
        for label in self.data_groups[group]['labels']:
            scores.extend(self.data_groups[group]['labels'][label]['scores'])
        if not scores:
            return 0.0
        return np.mean(scores)

    def max_group_score(self, group):
        scores = []
        for label in self.data_groups[group]['labels']:
            scores.extend(self.data_groups[group]['labels'][label]['scores'])
        if not scores:
            return 0.0
        return np.max(scores)

    def mean_label_score(self, group, label):
        if (label not in self.data_groups[group]['labels'] or
                not self.data_groups[group]['labels'][label]['scores']):
            return 0
        return np.mean(self.data_groups[group]['labels'][label]['scores'])

    def max_label_score(self, group, label):
        if (label not in self.data_groups[group]['labels'] or
                not self.data_groups[group]['labels'][label]['scores']):
            return 0
        return np.max(self.data_groups[group]['labels'][label]['scores'])

    def remove_group(self, group):
        for label in self.data_groups[group]['labels']:
            self.rejected.add(label)
            self.label_data_mapping[label].remove(group)
            if len(self.label_data_mapping[label]) == 0:
                del self.label_data_mapping[label]
        del self.data_groups[group]

    def reject(self, label):
        self.rejected.add(label)


class PredictionProcessor():
    def __init__(self, tc_preds, naive_preds, df, groups, abb_def_mappings):
        self.tc_preds = tc_preds  # textcat predictions
        self.naive_preds = naive_preds  # naive 'candidate dataset' matches
        self.df = df
        self.groups = groups
        self.abb_def_mappings = abb_def_mappings
        self.ds = Dataset(self.groups)
        self.connecting_words = [
            'of', 'in', 'on', 'and', 'for', 'from', 'the', 'in', 'a',
            'to', 'after', 'with', 'at', 'by', '&'
        ]
        self.cand_scores = defaultdict(list)

    def _get_thresholds(self, group):
        databases = ['database', 'data base', 'dataset',
                     'data set', 'data', 'data system']
        if not group:
            return [0.95, 0.4]
        if '(' in group:
            title = group[:group.rfind('(')].strip().lower()
        else:
            title = group.lower()
        if (any([title.endswith(database) for database in databases]) or
            any([f'{database} for' in title for database in databases]) or
                any([f'{database} of' in title for database in databases])):
            return [0.05, 0.05]
        if group in self.ds.data_groups:
            return [0.85, 0.25]
        return [0.95, 0.4]

    def _fill_cand_scores(self):
        for item in self.tc_preds:
            candidate = item[1]['candidate']
            clean_candidate = clean_label(candidate)
            self.cand_scores[clean_candidate].append(item[1]['score'])

    def remove_naive_preds_sentences(self):
        indices_to_remove = set()
        for i, item in enumerate(self.tc_preds):
            sentence = item[0]
            if '@cand@ dataset' in sentence.lower():
                indices_to_remove.add(i)

        self.tc_preds = [item for i, item in enumerate(self.tc_preds)
                         if i not in indices_to_remove]

    def group_naive(self):
        for Id, items in self.naive_preds.items():
            for item in items:
                sentence = item[0]
                for ent in item[1]['entities']:
                    cand = sentence[ent[0]:ent[1]]
                    cand_base = ' '.join(cand.split()[:-1])
                    clean_cand = clean_label(cand)
                    clean_cand_base = clean_label(cand_base)
                    if len(cand_base.split()) > 1:
                        if clean_cand in self.ds.label_group_mapping:
                            cand_groups = sorted(self.ds.label_group_mapping[clean_cand],
                                                 reverse=True,
                                                 key=lambda x: len(self.ds.label_group_mapping[x]))
                            group = cand_groups[0]
                            self.ds.add_label(group, cand, Id, 2)
                        elif clean_cand_base in self.ds.label_group_mapping:
                            cand_groups = sorted(self.ds.label_group_mapping[clean_cand_base],
                                                 reverse=True,
                                                 key=lambda x: len(self.ds.label_group_mapping[x]))
                            group = cand_groups[0]
                            self.ds.add_label(group, cand_base, Id, 2)
                    else:
                        if Id in self.abb_def_mappings and cand_base in self.abb_def_mappings[Id]:
                            for definition in self.abb_def_mappings[Id][cand_base]:
                                if clean_label(definition) in self.ds.label_group_mapping:
                                    cand_groups = sorted(self.ds.label_group_mapping[clean_label(definition)],
                                                         reverse=True,
                                                         key=lambda x: len(self.ds.label_group_mapping[x]))
                                    group = cand_groups[0]
                                    self.ds.add_label(group, cand_base, Id, 2)
                                    break
                        elif (len(cand_base) > 3 and cand_base == cand_base.upper() and
                              clean_cand_base in self.ds.label_group_mapping and
                              len(self.ds.label_group_mapping[clean_cand_base]) == 1):
                            group = self.ds.label_group_mapping[clean_cand_base][0]
                            self.ds.add_label(group, cand_base, Id, 1)

        log.info(
            f'Number of dataset groups after processing the naive predictions: {len(self.ds.data_groups)}'
        )

        # Remove titles containing lower case words
        for group in self.ds.data_groups.copy():
            if any([(w == w.lower() and w not in self.connecting_words and
                     w != 'dataset' and not re.search(r'[0-9]', w))
                    for w in group.split()]):
                self.ds.remove_group(group)
        log.info(
            f'Number of dataset groups after removing lower-case groups: {len(self.ds.data_groups)}'
        )

    def filter_preds(self):
        # Fill in the scores
        self._fill_cand_scores()
        cands_to_remove = set()
        for cand, v in self.cand_scores.items():
            clean_cand = clean_label(cand)
            if len(cand.split()) > 1:
                cand_groups = sorted(self.ds.label_group_mapping[clean_cand],
                                     reverse=True,
                                     key=lambda x: len(self.ds.label_group_mapping[x]))
                if cand_groups:
                    group = cand_groups[0]
                else:
                    group = None
            else:
                group = None
            max_threshold, mean_threshold = self._get_thresholds(group)
            if np.max(v) < max_threshold or np.mean(v) < mean_threshold:
                cands_to_remove.add(cand)

        indices_to_remove = set()
        for i, item in enumerate(self.tc_preds):
            cand = item[1]['candidate']
            clean_cand = clean_label(cand)
            if clean_cand in cands_to_remove:
                indices_to_remove.add(i)

        self.tc_preds = [item for i, item in enumerate(self.tc_preds)
                         if i not in indices_to_remove]
        log.info(
            f'Number of remaining sentences after filtering low scores: {len(self.tc_preds)}'
        )

    def group_long_candidates(self):
        indices_to_remove = set()
        for i, item in enumerate(self.tc_preds):
            Id = item[1]['Id']
            score = item[1]['score']
            cand = item[1]['candidate']
            clean_cand = clean_label(cand)
            if len(cand.split()) == 1:
                continue
            cand_groups = sorted(self.ds.label_group_mapping[clean_cand],
                                 reverse=True,
                                 key=lambda x: len(self.ds.label_group_mapping[x]))
            group = cand_groups[0]
            self.ds.add_label(group, cand, Id, score)
            indices_to_remove.add(i)
        self.tc_preds = [item for i, item in enumerate(self.tc_preds)
                         if i not in indices_to_remove]
        log.info(
            f'Number of dataset groups after grouping long candidates: {len(self.ds.data_groups)}'
        )
        log.info(
            f'Number of remaining sentences after grouping long candidates: {len(self.tc_preds)}'
        )

    def _find_matching_label(self, label, abbreviation):
        '''Finds a matching label that the label startwith
        Example: label='National Education Longitudinal Study of 1988'
                 matching_label = 'National Education Longitudinal Study'
        '''
        cleaned_label = clean_label(label)
        candidates = set()
        if abbreviation not in self.ds.label_data_mapping:
            return None
        else:
            for group in self.ds.label_data_mapping[abbreviation]:
                for cand in self.ds.data_groups[group]['labels']:
                    if len(cand.split()) == 1:
                        continue
                    if (cleaned_label.startswith(clean_label(cand)) and
                            len(cand.split()) >= len(abbreviation)):
                        candidates.add(cand)
        if candidates:
            return sorted(list(candidates), reverse=True,
                          key=lambda x: len(x))[0]
        return None

    def group_abbreviations(self):
        indices_to_remove = set()
        for i, item in enumerate(self.tc_preds):
            Id = item[1]['Id']
            score = item[1]['score']
            cand = item[1]['candidate']
            if len(cand.split()) != 1:
                continue
            if cand not in self.abb_def_mappings[Id]:
                continue
            for definition in self.abb_def_mappings[Id][cand]:
                clean_definition = clean_label(definition)
                if clean_definition in self.ds.label_group_mapping:
                    cand_groups = sorted(self.ds.label_group_mapping[clean_definition],
                                         reverse=True,
                                         key=lambda x: len(self.ds.label_group_mapping[x]))
                    group = cand_groups[0]
                    if group in self.ds.data_groups:
                        self.ds.add_label(group, cand, Id, score)
                    indices_to_remove.add(i)
                    break
        self.tc_preds = [item for i, item in enumerate(self.tc_preds)
                         if i not in indices_to_remove]

        indices_to_remove = set()
        for i, item in enumerate(self.tc_preds):
            Id = item[1]['Id']
            score = item[1]['score']
            cand = item[1]['candidate']
            if len(cand.split()) != 1:
                continue
            if cand not in self.abb_def_mappings[Id]:
                continue
            for definition in self.abb_def_mappings[Id][cand]:
                clean_definition = clean_label(definition)
                if clean_definition not in self.ds.label_group_mapping:
                    matched = self._find_matching_label(definition, cand)
                    if matched and clean_label(matched) in self.ds.label_group_mapping:
                        cand_groups = sorted(self.ds.label_group_mapping[clean_label(matched)],
                                             reverse=True,
                                             key=lambda x: len(self.ds.label_group_mapping[x]))
                        group = cand_groups[0]
                        if group in self.ds.data_groups:
                            self.ds.add_label(group, cand, Id, score)
                        indices_to_remove.add(i)
                        break

        self.tc_preds = [item for i, item in enumerate(self.tc_preds)
                         if i not in indices_to_remove]
        log.info(
            f'Number of dataset groups after grouping abbreviations: {len(self.ds.data_groups)}'
        )
        log.info(
            f'Number of remaining sentences after grouping abbreviations: {len(self.tc_preds)}'
        )

    def group_remaining(self):
        indices_to_remove = set()

        for i, item in enumerate(self.tc_preds):
            Id = item[1]['Id']
            score = item[1]['score']
            cand = item[1]['candidate']
            clean_cand = clean_label(cand)
            if clean_cand in self.ds.label_data_mapping:
                group = self.ds.label_data_mapping[clean_cand][0]
                self.ds.add_label(group, cand, Id, score)
            indices_to_remove.add(i)

        self.tc_preds = [item for i, item in enumerate(self.tc_preds)
                         if i not in indices_to_remove]
        log.info(
            f'Number of dataset groups after final grouping: {len(self.ds.data_groups)}'
        )
        log.info(
            f'Number of remaining sentences after final grouping: {len(self.tc_preds)}'
        )

    def filter_groups(self):

        # Filter out organizations
        for group in self.ds.data_groups.copy():
            if '(' in group:
                title = group[:group.rfind('(')].strip().lower()
            else:
                title = group.lower()
            if is_org_like(title):
                self.ds.remove_group(group)
        log.info(
            f'Number of dataset groups after removing organizations: {len(self.ds.data_groups)}'
        )

        # Remove short candidates with lower-case words or containing more than
        # one occurencies of data keywords, e.g. 'survey dataset'
        data_keywords = ['survey', 'study', 'initiative', 'program',
                         'programme', 'assessment', 'database', 'data base',
                         'data set', 'dataset', 'data']
        for group in self.ds.data_groups.copy():
            if '(' in group:
                title = group[:group.rfind('(')].strip()
                abb = group[group.rfind('(')+1:-1].strip()
            else:
                title = group
                abb = None
            if len(title.split()) < 3:
                if any([w == w.lower() for w in title.split()]):
                    self.ds.remove_group(group)
                elif len([w for w in title.lower().split()
                          if w in data_keywords]) > 1:
                    self.ds.remove_group(group)
                elif title == title.upper() or (abb and len(abb) < 3):
                    self.ds.remove_group(group)
        log.info(
            f'Number of dataset groups after cleaning short candidates: {len(self.ds.data_groups)}'
        )

        # Remove groups with some selected 'non-data' keywords
        nondata_keywords = ['method', 'example', 'resonance', 'tool',
                            'agreement', 'procedure', 'builder']
        reject_keywords = ['test', 'sample', 'cohort', 'supplement',
                           'act', 'file', 'index', 'trial', 'protocol',
                           'instrument', 'form', 'conference',
                           'infrastructure', 'trials']
        for group in self.ds.data_groups.copy():
            if '(' in group:
                title = group[:group.rfind('(')].strip().lower()
                abb = group[group.rfind('(')+1:-1].strip()
            else:
                title = group.lower()
                abb = None
            if any([keyword in title for keyword in nondata_keywords]):
                self.ds.remove_group(group)
            elif any([title.endswith(keyword) for keyword in reject_keywords]):
                self.ds.remove_group(group)
        log.info(
            f'Number of dataset groups after removing non-data keywords: {len(self.ds.data_groups)}'
        )

        # Remove groups with only short candidates, e.g. only abbreviations
        for group in self.ds.data_groups.copy():
            if all(len(cand) < 10 for cand in self.ds.data_groups[group]['labels']):
                self.ds.remove_group(group)
        log.info(
            f'Number of dataset groups after removing short groups: {len(self.ds.data_groups)}'
        )

        # Remove groups that are similar to larger groups
        groups_to_remove = set()
        all_group_titles = sorted([group for group in self.ds.data_groups],
                                  reverse=True,
                                  key=lambda x: len(self.ds.data_groups[x]['labels']))
        for i in range(len(all_group_titles) - 1):
            try:
                group_1 = all_group_titles[i]
                if '(' in group_1:
                    abb1 = group_1[group_1.rfind('(')+1:-1].strip()
                else:
                    abb1 = None
                for j in range(i + 1, len(all_group_titles)):
                    group_2 = all_group_titles[j]
                    for label in self.ds.data_groups[group_2]['labels']:
                        if any([label.startswith(cand) and len(label.replace(cand, '')) > 2 and
                                len(label.replace(cand, '')) < 10 for cand in self.ds.data_groups[group_1]['labels']]):
                            if '(' in group_2:
                                abb2 = group_2[group_2.rfind('(')+1:-1].strip()
                                if len(abb2.split()) > 1 or re.search(r'[^A-Za-z]', abb2):
                                    groups_to_remove.add(group_2)
                                elif abb1 and not abb2.startswith(abb1) and re.search(r'[^A-Za-z]', abb2):
                                    groups_to_remove.add(title2)
                        elif any([label.startswith(cand) and len(label.replace(cand, '')) > 2
                                  for cand in self.ds.data_groups[group_1]['labels']]):
                            if '(' in group_2:
                                abb2 = group_2[group_2.rfind('(')+1:-1].strip()
                                if abb1 and abb2.startswith(abb1) and re.search(r'[^A-Za-z]', abb1):
                                    groups_to_remove.add(group_2)
            except:
                continue

        for group in groups_to_remove:
            self.ds.remove_group(group)
        log.info(
            f'Number of groups after removing smaller repeated groups: {len(self.ds.data_groups)}'
        )

        # Final filtering: Remove groups that are not data like and contain
        # special characters or lower-case words in the title

        for group in self.ds.data_groups.copy():
            if '(' in group:
                title = group[:group.rfind('(')].strip().lower()
                abb = group[group.rfind('(')+1:-1].strip()
            else:
                title = group.lower()
            if not is_data_like(title, list(self.ds.data_groups[group]['labels'].keys())):
                if abb and re.search('[^A-Z&a-z0-9]', abb):
                    self.ds.remove_group(group)
                elif any([w == w.lower() and w not in self.connecting_words and
                          not re.search(r'[0-9]', w) for w in group.split()]):
                    self.ds.remove_group(group)
                elif len(self.ds.data_groups[group]['Ids']) < 5:
                    self.ds.remove_group(group)
                elif title.endswith('system'):
                    self.ds.remove_group(group)

        log.info(
            f'Final Number of groups: {len(self.ds.data_groups)}'
        )

    def construct_final_labels(self):
        # Get the ids for each group and label
        group_ids = defaultdict(set)
        for group in self.ds.data_groups:
            for label in self.ds.data_groups[group]['labels']:
                group_ids[group].update(
                    self.ds.data_groups[group]['labels'][label]['Ids']
                )

        id_clean_text = self.df.groupby('Id')['clean_text'].agg(list)
        group_label_counter_found = {}  # count occurencies found in 'data' sentences
        group_label_counter_all = {}  # count overall occurencies
        for group in self.ds.data_groups:
            group_label_counter_found[group] = Counter()
            group_label_counter_all[group] = Counter()
            for row_id in self.df['Id'].unique():
                cleaned_text = id_clean_text[row_id][0]
                for label in {clean_label(l) for l in self.ds.data_groups[group]['labels']}:
                    if f' {label.strip()} ' in cleaned_text:
                        if row_id in group_ids[group]:
                            group_label_counter_found[group][label] += 1
                        group_label_counter_all[group][label] += 1

        # Remove labels that occur much often then found, e.g. AGE
        for group in self.ds.data_groups.copy():
            for label, count_all in group_label_counter_all[group].items():
                if (group in group_label_counter_found and
                        label in group_label_counter_found[group]):
                    count_found = group_label_counter_found[group][label]
                    if len(label) < 8 and count_found < count_all / 4:
                        for cand in self.ds.data_groups[group]['labels'].copy():
                            if clean_label(cand) == label:
                                try:
                                    del self.ds.data_groups[group]['labels'][cand]
                                except KeyError:
                                    continue
                        if len(self.ds.data_groups[group]['labels']) == 0:
                            self.ds.remove_group(group)
                        elif all([len(cand) < 6 for cand in self.ds.data_groups[group]['labels']]):
                            self.ds.remove_group(group)
                    elif len(label) >= 8 and count_found < count_all / 20:
                        for cand in self.ds.data_groups[group]['labels'].copy():
                            if clean_label(cand) == label:
                                try:
                                    del self.ds.data_groups[group]['labels'][cand]
                                except KeyError:
                                    continue
                        if len(self.ds.data_groups[group]['labels']) == 0:
                            self.ds.remove_group(group)
                        elif all([len(cand) < 6 for cand in self.ds.data_groups[group]['labels']]):
                            self.ds.remove_group(group)

        # Count the number of times when the label is the only one in the group
        # that appears. Important for deciding whether to include the abbreviations
        # in the final list of labbels
        id_clean_text = self.df.groupby('Id')['clean_text'].agg(list)
        group_label_single_counter = {}
        for group in self.ds.data_groups:
            group_label_single_counter[group] = Counter()
            for row_id in self.df['Id'].unique():
                cleaned_text = id_clean_text[row_id][0]
                for label in sorted(list({clean_label(l) for l in self.ds.data_groups[group]['labels']}),
                                    reverse=True, key=lambda x: len(x)):
                    if f' {label.strip()} ' in cleaned_text:
                        group_label_single_counter[group][label] += 1
                        break

        # Decide whether to keep the appreviations or not
        included_abbreviations = set()
        group_label_frequencies = {}
        for group in group_label_single_counter:
            group_label_frequencies[group] = {}
            count = sum([v for v in group_label_single_counter[group].values()])
            if count == 0:
                continue
            for label in group_label_single_counter[group]:
                group_label_frequencies[group][label] = group_label_single_counter[group][label] / count
                if len(label) < 8 and len(label) > 3:
                    abb_freq = group_label_frequencies[group][label]
                    if len(label) < 5:
                        if abb_freq > 0.15 and abb_freq < 0.25:
                            included_abbreviations.add(label)
                    else:
                        if abb_freq > 0.01 and abb_freq < 0.35:
                            included_abbreviations.add(label)

        # Final list of labels per group sorted by the frequency
        final_group_labels = {}
        for group in self.ds.data_groups:
            try:
                if group not in group_label_counter_all:
                    continue
                final_group_labels[group] = {'abbreviation': None,
                                             'most_common': None,
                                             'remaining': []}
                all_labels = [clean_label(label) for label in self.ds.data_groups[group]['labels']
                              if (clean_label(label) in group_label_counter_all[group] and
                                  clean_label(label) in group_label_single_counter[group] and
                                  group_label_frequencies[group][clean_label(label)] > 0.005 and
                                  group_label_single_counter[group][clean_label(label)] > 1)]
                for label in all_labels.copy():
                    if len(label) < 8 and label in included_abbreviations:
                        final_group_labels[group]['abbreviation'] = label
                        all_labels.remove(label)
                    elif len(label) < 8:
                        all_labels.remove(label)

                try:
                    all_labels = sorted(set(all_labels), reverse=True,
                                        key=lambda x: group_label_counter_all[group][x])
                    final_group_labels[group]['most_common'] = all_labels[0]
                    final_group_labels[group]['remaining'] = all_labels[1:]
                    if len(final_group_labels[group]['remaining']) != 1:
                        for label in final_group_labels[group]['remaining'].copy():
                            if (label.startswith(final_group_labels[group]['most_common']) and
                                    len(label.split()) == len(final_group_labels[group]['most_common'].split()) + 1):
                                final_group_labels[group]['remaining'].remove(label)
                except:
                    continue
            except:
                continue

        return final_group_labels
