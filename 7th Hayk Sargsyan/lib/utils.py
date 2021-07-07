import json
import pandas as pd
from collections import defaultdict


def load_settings(settings_path):
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    return settings


def load_test_data(settings):
    df = pd.read_csv(settings['TEST_DATA_CLEAN_PATH'])
    with open(settings['TEST_TEXTCAT_DATA_PATH'], 'r') as f:
        textcat_data = json.load(f)
    with open(settings['TEST_NAIVE_PREDICTIONS_DATA_PATH'], 'r') as f:
        naive_preds = json.load(f)
    with open(settings['TEST_DATA_GROUP_PATH'], 'r') as f:
        groups = json.load(f)
    with open(settings['TEST_ABB_DEF_MAPPINGS_PATH'], 'r') as f:
        abb_def_mappings = json.load(f)
    return (df, textcat_data, groups, abb_def_mappings, naive_preds)


def is_group_unclear(title):
    if len(title.split()) < 3:
        return False
    if title == 'general circulation models':
        return True
    keywords = ['survey', 'study', 'initiative', 'program', 'programme',
                'assessment', 'database', 'data base', 'data set',
                'surveys', 'studies', 'dataset', 'data', 'model']
    for key in keywords:
        if title.lower().endswith(key):
            if ('institute of' in title.lower() or
                'association of' in title.lower() or
                'institute for' in title.lower() or
                'association for' in title.lower() or
                'institute on' in title.lower() or
                    'association on' in title.lower()):
                return False
            return True
        if (f'{key} of ' in title.lower() or f'{key} on ' in title.lower() or
                f'{key} for ' in title.lower()):
            if ('institute of' in title.lower() or
                'association of' in title.lower() or
                'institute for' in title.lower() or
                'association for' in title.lower() or
                'institute on' in title.lower() or
                    'association on' in title.lower()):
                return False
            return True
    return False


def get_title_label_mapping(df):
    mapping = df.groupby('dataset_title')['dataset_label'].agg(set).agg(list)
    for k, v in mapping.to_dict().items():
        mapping[k] = sorted(v, reverse=True, key=lambda x: len(x))
    return mapping


def get_candidate_group_mapping(groups):

    candidate_group_mapping = defaultdict(list)
    for k, v in groups.items():
        for cand in v:
            if k not in candidate_group_mapping[cand]:
                candidate_group_mapping[cand].append(k)
    return candidate_group_mapping


def get_dataset_groups(df, groups):
    candidate_group_mapping = get_candidate_group_mapping(groups)
    title_labels = get_title_label_mapping(df)

    dataset_groups = defaultdict(set)
    for k, v in title_labels.items():
        for cand in v:
            if cand.endswith(')'):
                cand = cand[:-1]
            if cand.startswith('the ') or cand.startswith('The '):
                cand = cand[4:]
            if cand in candidate_group_mapping:
                dataset_groups[cand].add(candidate_group_mapping[cand][0])

    return dataset_groups


def is_dataset_like(title, cands):
    if len(title.split()) < 3:
        return False
    keywords = ['survey', 'study', 'initiative', 'program', 'programme',
                'assessment', 'database', 'data base', 'data set',
                'dataset', 'data']
    for key in keywords:
        if title.lower().endswith(key):
            if ('institute of' in title.lower() or
                'association of' in title.lower() or
                'institute for' in title.lower() or
                'association for' in title.lower() or
                    'institute on' in title.lower() or
                    'association on' in title.lower()):
                return False
            return True
        if (f'{key} of ' in title.lower() or
                f'{key} on ' in title.lower() or
                f'{key} for ' in title.lower()):
            if ('institute of' in title.lower() or
                'association of' in title.lower() or
                'institute for' in title.lower() or
                'association for' in title.lower() or
                    'institute on' in title.lower() or
                    'association on' in title.lower()):
                return False
            return True
    for key in keywords:
        if any([cand.lower().endswith(f' key') for cand in cands]):
            return True
    return False


def is_data_like(title, cands):
    keywords = ['survey', 'study', 'initiative', 'program', 'programme',
                'inventory', 'assessment', 'model', 'network', 'sequence',
                'practice', 'project', 'datum', 'database', 'data base',
                'data set', 'list', 'archive', 'interpolation', 'atlas',
                'surveys', 'studies', 'dataset', 'data', 'model', 'registry',
                'census', 'encyclopedia']
    if any([key in title.lower() for key in keywords]):
        return True
    for key in keywords:
        if any([cand.lower().endswith(key) for cand in cands]):
            return True
    return False


def is_org_like(candidate):
    if len(candidate.split()) == 1:
        return False
    if 'data' in candidate and not candidate.endswith('center'):
        return False
    keywords = ['institute', 'institute', 'center', 'foundation',
                'organisation', 'administration', 'organizations',
                'alliance', 'clinics', 'institut', 'institutes', 'society',
                'centers', 'unit', 'collaboration', 'bureau', 'university',
                'service', 'department', 'divisiion', 'agency',
                'office', 'library', 'organization', 'board', 'council',
                'union', 'college', 'committee', 'consortium',
                'association', 'clinic', 'hospital', 'laboratory',
                'centre', 'ministry', 'panel', 'school', 'schools',
                'facility', 'commission', 'league', 'taskforce',
                'register', 'insurance']

    if any([candidate.endswith(keyword) for keyword in keywords]):
        if not is_dataset_like(candidate.lower(), [candidate.lower()]):
            return True
        else:
            return False
    if any([f'{keyword} of' in candidate for keyword in keywords]):
        if not is_dataset_like(candidate.lower(), [candidate.lower()]):
            return True
        else:
            return False
    if any([f'{keyword} on' in candidate for keyword in keywords]):
        if not is_dataset_like(candidate.lower(), [candidate.lower()]):
            return True
    if any([f'{keyword} for' in candidate for keyword in keywords]):
        if not is_dataset_like(candidate.lower(), [candidate.lower()]):
            return True
    return False
