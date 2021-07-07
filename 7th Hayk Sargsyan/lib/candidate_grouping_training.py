import re
import logging
from collections import Counter, defaultdict

log = logging.getLogger(__name__)

connecting_words = ["of", "in", "on", "and", "for", "from",
                    "the", "in", "a", "to", "after", "with", "at", "by"]


def clean_label(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())


def get_abbreviation(key):
    return key.split("(")[-1][:-1].strip()


def get_definition(key):
    return key.split("(")[0].strip()


def clean_text(text):
    text = re.sub(r"[^a-z\']", " ", text.lower())
    text = re.sub(r"[\']", "", text.lower())
    text = re.sub(r"[\)]", "", text.lower())
    text = re.sub(r"[\(]", "", text.lower())
    return text


def separate(str1, str2):
    institutions = ["centre", "centres", "institute", "institutes",
                    "service", "services", "foundation", "foundations",
                    "association", "associations"]
    studies = ["study", "survey", "studies", "surveys",
               "initiative", "initiatives", "program", "programs"]
    if str1.split()[-1].lower() in institutions and str2.split()[-1].lower() in studies:
        return True
    if str2.split()[-1].lower() in institutions and str1.split()[-1].lower() in studies:
        return True
    return False


def is_word_overlap(str1, str2, threshold=0.5, ignore_connecting_words=True):
    if (min(len(str1.split()), len(str2.split())) > 1 and
            (str1.lower().startswith(str2.lower()) or str2.lower().startswith(str1.lower()))):
        return True
    str1_mod = str1
    str2_mod = str2
    if ignore_connecting_words:
        for w in connecting_words:
            str1_mod = str1_mod.replace(f" {w} ", " ")
            str2_mod = str2_mod.replace(f" {w} ", " ")

    str1_mod = str1_mod.replace("'s", "s")
    str2_mod = str2_mod.replace("'s", "s")
    str1_mod = re.sub(r"[\,.:;]", "", str1_mod)
    str2_mod = re.sub(r"[\,.:;]", "", str2_mod)
    a = set(clean_text(str1_mod).split())
    b = set(clean_text(str2_mod).split())
    c = a.intersection(b)
    word_overlap = float(len(c)) / (max(len(a), len(b)))
    if word_overlap > threshold:
        return True
    return False


def is_char_overlap(str1, str2, threshold=0.2):
    a_counter = Counter("".join(clean_text(str1).split()))
    b_counter = Counter("".join(clean_text(str2).split()))
    c_counter = Counter({key: b_counter.get(key, 0) -
                         value for key, value in a_counter.items()})
    c_counter_sum = sum([abs(val) for val in c_counter.values()])
    if abs(c_counter_sum / sum(a_counter.values())) < threshold:
        return True
    # Try with a substring before ","
    str2_mod = str2.split(",")[0]
    b_counter = Counter("".join(clean_text(str2_mod).split()))
    c_counter = Counter({key: b_counter.get(key, 0) -
                         value for key, value in a_counter.items()})
    c_counter_sum = sum([abs(val) for val in c_counter.values()])
    if abs(c_counter_sum / sum(a_counter.values())) < threshold:
        return True
    return False


def overlap(str1, str2, abb):
    if is_word_overlap(str1, str2):
        return True

    if (any([word.replace(".", "") in abb for word in str2.split()]) and
            len(abb.split("-")) == 1 and len(abb.split()) == 1 and len(abb.split("/")) == 1):
        sub_abb_word = [word for word in str2.split()
                        if word.replace(".", "") in abb][0]
        str2_mod = str2.replace(sub_abb_word, "").strip()
        if is_word_overlap(str1, str2_mod, threshold=0.3, ignore_connecting_words=False):
            return True

    if is_char_overlap(str1, str2):
        return True

    return False


def exclude_candidate_with_special_symbols(candidate):
    """Excludes candidates that contain any of the exclusion symbols"""
    exclusion_symbols = "[]()%="
    if any([exclusion_symbol in candidate for exclusion_symbol in exclusion_symbols]):
        return True
    if re.search(r" and [0-9]+", candidate.lower()):
        return True
    return False


def exclude_candidate(candidate, abb, count):
    if exclude_candidate_with_special_symbols(candidate):
        return True
    if candidate.startswith("al "):
        return True
    for w in connecting_words:
        if candidate.endswith(f" {w}"):
            return True
        if w == "the":
            continue
        if candidate.startswith(f"{w} "):
            return True
    if "i.e" in candidate:
        return True
    if count < 3:
        if any([w in abb or abb in w for w in candidate.split()]):
            return False
        cleaned_candidate = clean_text(candidate)
        if cleaned_candidate.split()[0][0].lower() != abb[0].lower():
            return True

    return False


def all_words_capital(text):
    return all([word[0] == word[0].upper() for word in text.split()])


def all_words_capital(text):
    return all([word[0] == word[0].upper() for word in text.split()])


def has_pascal_pattern(term):
    term_mod = term.replace('"', "")
    for word in connecting_words:
        term_mod = term_mod.replace(f" {word} ", f" {word.upper()} ")
        term_mod = term_mod.replace(f"-{word}-", f"-{word.upper()}-")
        term_split = term_mod.split("-")
        term_full_split = []
        for t in term_split:
            term_full_split.extend(t.split())
        if all(re.search(r"[A-Z0-9&]", w[0]) for w in term_full_split if w != ""):
            return True

    if (len(term_mod.split()) > 4 and
            len([w for w in term_mod.split() if re.search(r"[A-Z0-9&]", w[0])]) == len(term_mod.split()) - 1):
        return True

    return False


def exclude_group(terms, title, exclude_lower_case):
    abbreviation = get_abbreviation(title)
    definition = get_definition(title)
    if exclude_lower_case and definition == definition.lower():
        return True
    if re.sub(r"[^A-Za-z]", "", abbreviation).strip() in definition and re.search(r"[0-9]", definition):
        return True
    if "data" in abbreviation.lower() and abbreviation.lower() != "data":
        return False
    if len([w for w in definition.split() if re.search(r"[A-Z]", w[0])]) >= 2:
        return False

    for term in terms:
        term_mod = get_definition(term)
        if has_pascal_pattern(term_mod):
            return False
        term_split = term_mod.split("-")
        term_full_split = []
        for t in term_split:
            term_full_split.extend(t.split())
        term_split = [w for w in term_full_split if w not in connecting_words]
        if len(term_split) == 1:
            continue
        if len(term_split) > len(abbreviation):
            term_split = term_split[:-1]
        if all([w[0].lower() in abbreviation.lower() for w in term_split]):
            return False
        if f"{term_mod[0].lower()}{' '.join(term_mod.split()[1:])}".lower() == abbreviation.lower():
            return False
    return True


def sanitize_candidate(candidate):
    candidate = candidate.strip()
    while candidate[-1] in ".,:_-;*":
        candidate = candidate[:-1].strip()
    return candidate


def group_for_same_abbreviation(abb, candidates, exclude_lower_case):
    groups = defaultdict(list)
    if "," in abb or exclude_candidate_with_special_symbols(abb):
        return groups

    most_common = list(reversed(Counter(candidates).most_common()))
    if not most_common:
        return groups
    candidate, count = most_common.pop()

    while most_common and exclude_candidate(candidate, abb, count):
        candidate, count = most_common.pop()
    if exclude_candidate(candidate, abb, count):
        return groups

    # clean candidate
    while candidate.startswith("-"):
        candidate = candidate[1:]
    dataset_title = f"{sanitize_candidate(candidate)} ({sanitize_candidate(abb)})"
    groups[dataset_title].append(sanitize_candidate(candidate))
    groups[dataset_title].append(
        f"{sanitize_candidate(candidate)} ({sanitize_candidate(abb)})")

    while most_common:
        candidate, count = most_common.pop()
        while candidate.startswith("-"):
            candidate = candidate[1:]
        if exclude_candidate_with_special_symbols(candidate):
            continue
        for title in groups:
            to_separate = False
            for label in groups[title]:
                if separate(label, candidate):
                    to_separate = True
                if overlap(label, candidate, abb) and not to_separate:
                    groups[title].append(sanitize_candidate(candidate))
                    groups[title].append(
                        f"{sanitize_candidate(candidate)} ({sanitize_candidate(abb)})")
                    break
            if sanitize_candidate(candidate) in groups[title]:
                break
        else:
            if not exclude_candidate(candidate, abb, count):
                dataset_title = f"{sanitize_candidate(candidate)} ({sanitize_candidate(abb)})"
                groups[dataset_title].append(sanitize_candidate(candidate))
                groups[dataset_title].append(
                    f"{sanitize_candidate(candidate)} ({sanitize_candidate(abb)})")
    for title in groups.copy():
        if exclude_group(groups[title], title, exclude_lower_case):
            del groups[title]
    return groups


def group_train_candidates(candidates, sentences_candidates,
                           title_labels, exclude_lower_case=False):
    # remove rare and/or weird abbreviations that have the definition elsewhere
    mapping = defaultdict(list)
    for k, v in candidates.items():
        for cand in v:
            if k not in mapping[clean_label(cand)]:
                mapping[clean_label(cand)].append(k)
    for k in mapping:
        if len(mapping[k]) == 1 or len({v.lower() for v in mapping[k]}) == 1:
            continue
        for w in mapping[k]:
            if sum([v for v in candidates[w].values()]) == 1:
                del candidates[w]

    dataset = defaultdict(list)
    for abb in candidates:
        group = group_for_same_abbreviation(
            abb, candidates[abb], exclude_lower_case)
        if len(group) >= 1:
            dataset.update(group)

    # Exclude short candidates
    for k, v in dataset.copy().items():
        abb = get_abbreviation(k)
        if re.search(r"[0-9]", abb[-1]):
            continue
        abb = re.sub(r"[^a-zA-Z]", "", abb)
        to_break = True
        for cand in v[:]:
            orig_cand = cand
            if cand.split("(")[0].strip().split()[-1][0].lower() == abb[-1].lower():
                to_break = False
            if to_break:
                continue
            if "(" in cand:
                cand = cand[:cand.rfind("(")].strip()
            clean_cand = cand.replace("-", " ").replace("/", " ")
            if any([w in abb for w in clean_cand.split()]):
                continue
            if any([w in abb for w in cand.split()]):
                continue
            if (clean_cand.split()[-1][0].lower() != abb[-1].lower() and
                    cand.split()[-1][0].lower() != abb[-1].lower()):
                if clean_cand.split()[-1].lower() not in ["study", "survey", "model", "initiative", "program"]:
                    dataset[k].remove(orig_cand)

    # Remove groups where candidate and definition are the same
    for k in dataset.copy():
        abb = get_abbreviation(k)
        definition = k.replace(f"({abb})", "").strip()
        if abb.lower() == definition.lower():
            del dataset[k]

    keys_to_remove = set()
    for k1 in dataset:
        abb1 = get_abbreviation(k1)
        long1 = k1.replace(f"({abb1})", "").strip()
        if len(abb1.split()) == 1 and len(abb1.split("-")) == 1:
            continue
        for k2 in dataset:
            abb2 = get_abbreviation(k2)
            if abb2 == abb1:
                continue
            long2 = k2.replace(f"({abb2})", "").strip()
            if abb1.startswith(abb2):
                diff = abb1.replace(abb2, "").strip().replace("-", "")
                if not re.search(r"[^A-Za-z]", diff) and diff == diff.upper():
                    continue
                if long1 == f"{long2} {diff}":
                    keys_to_remove.add(k1)
                elif long1 == f"{long2}-{diff}":
                    keys_to_remove.add(k1)
            if abb1.endswith(abb2):
                diff = abb1.replace(abb2, "").strip().replace("-", "")
                if not re.search(r"[^A-Za-z]", diff) and diff == diff.upper():
                    continue
                if long1 == f"{diff} {long2}":
                    keys_to_remove.add(k1)
                elif long1 == f"{diff}-{long2}":
                    keys_to_remove.add(k1)

    for k in keys_to_remove:
        if k in dataset:
            del dataset[k]

    keys_to_remove = set()
    for k1 in dataset:
        abb1 = get_abbreviation(k1)
        long1 = k1.replace(f"({abb1})", "").strip()
        if not re.search(r"[0-9]", abb1):
            continue
        for k2 in dataset:
            abb2 = get_abbreviation(k2)
            if abb2 == abb1:
                continue
            if abb2 in abb1 and is_word_overlap(k2, k1):
                diff = abb1.replace(abb2, "")
                if re.search(r"[0-9\-]", diff):
                    keys_to_remove.add(k1)

    for k in keys_to_remove:
        if k in dataset:
            del dataset[k]

    for k, v in dataset.copy().items():
        remove_numbers = False
        if not re.search(r"[0-9]", v[0]):
            remove_numbers = True
        else:
            for cand in v[1:]:
                if "(" in cand:
                    continue
                if v[0].lower().startswith(f"{cand.lower()}") and not re.search(r"[0-9]", cand):
                    remove_numbers = True
                break
        if remove_numbers:
            for cand in v[:]:
                if re.search(r"\s[0-9]", cand):
                    v.remove(cand)
        if not "cohort" in v[0].lower():
            for cand in v[:]:
                if "cohort" in cand.lower():
                    v.remove(cand)
        if not "class" in v[0].lower():
            for cand in v[:]:
                if "class" in cand.lower():
                    v.remove(cand)

    # rename groups
    keys_to_remove = set()
    for k in dataset.copy():
        definition = k.split("(")[0].strip()
        if definition not in dataset[k]:
            for cand in dataset[k]:
                if "(" in cand:
                    if cand in dataset:
                        dataset[cand].extend(
                            [c for c in dataset[k] if c not in dataset[cand]])
                        keys_to_remove.add(k)
                    else:
                        dataset[cand] = [item for item in dataset[k]]
                        keys_to_remove.add(k)
                        break

    for k in keys_to_remove:
        if k in dataset:
            del dataset[k]

    # remove single word groups
    for k in dataset.copy():
        definition = k.split("(")[0].strip()
        if (len(definition.split()) == 1 and not re.search(r"\-", definition) and
                len([c for c in definition if c == c.upper()]) < 2):
            del dataset[k]

    to_be_excluded_groups = []
    to_be_excluded_cands = defaultdict(list)
    excluded_abbreviations = []
    pattern1 = re.compile(r"[\-\_\:\s]*[0-9]+")
    pattern2 = re.compile(r"[\-\_\:\s]*[A-Za-z]{1}")
    for k1, v1 in dataset.items():
        abb1 = k1.split("(")[1].strip()[:-1]
        long1 = k1.split("(")[0].strip()
        if not re.search(r"[^A-Za-z]", abb1):
            continue
        for k2, v2 in dataset.items():
            abb2 = k2.split("(")[1].strip()[:-1]
            long2 = k2.split("(")[0].strip()
            if abb1 == abb2:
                continue
            if abb1.lower().startswith(abb2.lower()):
                diff = abb1.replace(abb2, "")
                skip = False
                for c in diff:
                    if re.search(r"[A-Z]", c) and c not in long1:
                        skip = True
                        break
                if skip:
                    continue
                diff = abb1.lower().replace(abb2.lower(), "")

                if ((pattern1.search(diff) and pattern1.search(diff).group(0) == diff) or
                        (pattern2.search(diff) and pattern2.search(diff).group(0) == diff)):
                    if any([long1.lower().startswith(cand.lower()) for cand in v2]):
                        for cand in v2:
                            if long1.lower().startswith(cand.lower()):
                                extra = long1.lower().replace(cand.lower(), "").strip()
                                if not re.search(r"[^A-za-z]", extra) and len(extra.split()) <= 1:
                                    break
                                to_be_excluded_cands[k2].append(
                                    re.sub(r"[^A-Za-z\s0-9]", "", extra))
                                break
                        if not re.search(r"[^A-za-z]", extra) and len(extra.split()) == 1:
                            continue
                        to_be_excluded_groups.append(k1)
                        excluded_abbreviations.append(abb1)

    for k1 in to_be_excluded_groups:
        abb1 = k1.split("(")[1].strip()[:-1]
        long1 = k1.split("(")[0].strip()
        new_abb = re.sub(r"[^A-Za-z0-9]+", "", abb1)
        for k2 in dataset:
            abb2 = k2.split("(")[1].strip()[:-1]
            if new_abb == abb2 and k2 not in to_be_excluded_groups:
                if any([is_word_overlap(long1, cand.split("(")[0].strip()) for cand in dataset[k2]]):
                    to_be_excluded_groups.append(k2)

    for k in to_be_excluded_groups:
        if k in dataset:
            del dataset[k]

    for abb1 in excluded_abbreviations:
        for k in dataset.copy():
            abb2 = k.split("(")[-1][:-1]
            if abb1.lower() == abb2.lower():
                del dataset[k]

    for k, v in to_be_excluded_cands.items():
        if k in dataset:
            removed = []
            for cand in v:
                for cand2 in dataset[k][:]:
                    if cand2.lower().endswith(cand.lower()) and cand2 in dataset[k]:
                        dataset[k].remove(cand2)
                        removed.append(cand2)
            for cand2 in dataset[k][:]:
                if any([r in cand2 for r in removed]) and cand2 in dataset[k]:
                    dataset[k].remove(cand2)

    to_be_removed = defaultdict(set)
    for k, v in dataset.items():
        for cand1 in v:
            if ("(") in cand1:
                abb1 = cand1.split("(")[1].strip()[:-1]
                long1 = cand1.split("(")[0].strip()
            else:
                long1 = cand1
            for cand2 in v:
                if cand1.lower() == cand2.lower():
                    continue
                if "(" in cand2:
                    abb2 = cand2.split("(")[1].strip()[:-1]
                    long2 = cand2.split("(")[0].strip()
                else:
                    long2 = cand2
                if long2.lower().startswith(long1.lower()):
                    diff = long2.lower().replace(long1.lower(), "")
                    if re.search(r"[0-9\:\-]", diff):
                        to_be_removed[k].add(cand2)

    for k in dataset:
        for cand in to_be_removed[k]:
            if cand in dataset[k]:
                dataset[k].remove(cand)

    def upper_count(text):
        return len([c for c in text if c == c.upper()])

    to_be_removed = defaultdict(set)
    for k, v in dataset.items():
        abb = k.split("(")[1].strip()[:-1]
        is_first_char_abb = False
        i = 0
        cand = dataset[k][0]
        while not is_first_char_abb and i < len(dataset[k]):
            cand = dataset[k][i]
            cand = cand.replace("-", " ")
            cand = cand.replace("/", " ")
            if ("(") in cand:
                definition = cand.split("(")[0].strip()
            else:
                definition = cand
            first_chars = "".join([w[0].lower() for w in cand.split()])
            if abb.lower() in first_chars:
                is_first_char_abb = True
            else:
                i += 1
        if is_first_char_abb:
            for j, cand in enumerate(v[i+1:]):
                cand = cand.replace("-", " ")
                cand = cand.replace("/", " ")
                if ("(") in cand:
                    definition = cand.split("(")[0].strip()
                else:
                    definition = cand
                first_chars = "".join([w[0].lower()
                                       for w in definition.split()])
                if abb.lower() not in first_chars and first_chars in abb.lower():
                    first_chars = "".join([w[0].lower() if upper_count(
                        w) < 2 else w.lower() for w in definition.split()])
                    if abb.lower() not in first_chars:
                        if not any([is_char_overlap(prev_cand, definition, threshold=0.1) for prev_cand in v[:j]
                                    if prev_cand not in to_be_removed[k] and "(" not in prev_cand]):
                            to_be_removed[k].add(cand)
                            to_be_removed[k].add(f"{cand} ({abb}")

    for k in dataset:
        abb = k.split("(")[1].strip()[:-1]
        for cand in to_be_removed[k].copy():
            if ("(") in cand:
                definition = cand.split("(")[0].strip()
            else:
                definition = cand
            for cand2 in dataset[k]:
                if ("(") in cand2:
                    long2 = cand2.split("(")[0].strip()
                else:
                    long2 = cand2
                if is_char_overlap(definition, long2, threshold=0.05):
                    to_be_removed[k].add(cand2)
                    to_be_removed[k].add(f"{cand2} ({abb}")

    for k in dataset:
        for cand in to_be_removed[k]:
            if cand in dataset[k]:
                dataset[k].remove(cand)

    not_remove = defaultdict(set)
    to_remove = defaultdict(set)
    for k in dataset:
        for i in range(len(dataset[k]) - 1):
            cand1 = dataset[k][i]
            if ("(") in cand1:
                abb1 = cand1.split("(")[1].strip()[:-1]
                long1 = cand1.split("(")[0].strip()
            else:
                long1 = cand1
            for j in range(i+1, len(dataset[k])):
                cand2 = dataset[k][j]
                if cand2 in not_remove[k] or cand2 in to_remove[k]:
                    continue
                if ("(") in cand2:
                    abb2 = cand2.split("(")[1].strip()[:-1]
                    long2 = cand2.split("(")[0].strip()
                else:
                    long2 = cand2
                if (long2.lower().endswith("study") or long2.lower().endswith("studies") or
                    long2.lower().endswith("survey") or long2.lower().endswith("surveys") or
                    long2.lower().endswith(" data") or long2.lower().endswith(" dataset") or
                    long2.lower().endswith(" database") or long2.lower().endswith(" program") or
                        long2.lower().endswith(" initiative") or long2.lower().endswith(" model")):
                    not_remove[k].add(cand2)
                    continue
                if long2.lower() == long1.lower():
                    not_remove[k].add(cand2)
                    continue
                if long2.lower().startswith(long1.lower()):
                    diff = long2.lower().replace(long1.lower(), "a")
                    if len(diff.split()) > 1:
                        to_remove[k].add(cand2)
                    else:
                        not_remove[k].add(cand2)

    for k in dataset:
        for cand in to_remove[k]:
            if cand in dataset[k]:
                dataset[k].remove(cand)

    def has_overlap(label_dataset):
        for k, v in label_dataset.items():
            if len(v) > 1:
                return True
        return False

    def merge_dataset(dataset, label_dataset):

        for k, v in label_dataset.items():
            for i in range(1, len(v)):
                if v[i] not in dataset:
                    continue
                if v[0] not in dataset:
                    continue
                dataset[v[0]].extend(
                    [c for c in dataset[v[i]] if c not in dataset[v[0]]])
                del dataset[v[i]]

        return dataset

    def expand_labels(dataset):
        label_dataset = defaultdict(list)
        for k, v in dataset.items():
            for candidate in v:
                cleaned_candidate = clean_label(candidate)
                if k not in label_dataset[cleaned_candidate]:
                    label_dataset[cleaned_candidate].append(k)
        return label_dataset

    def construct_label_dataset_mapping(dataset):
        label_dataset_mapping = {}
        for k, v in dataset.items():
            for label in v:
                cleaned_label = clean_label(label)
                if cleaned_label not in label_dataset_mapping:
                    label_dataset_mapping[cleaned_label] = {
                        "group_name": k,
                        "scores": []
                    }
        return label_dataset_mapping

    def is_regular_title(title):
        abb = re.sub(r"[^A-Za-z]", "", get_abbreviation(title))
        definition = get_definition(title)
        first_chars = [w[0].lower() for w in definition.split()]
        if "".join(first_chars) == abb.lower() or "".join(first_chars) == abb[:-1].lower():
            return True
        first_chars = [w[0].lower() for w in re.sub(
            r"[^A-Za-z]", " ", definition).split()]
        if "".join(first_chars) == abb.lower() or "".join(first_chars) == abb[:-1].lower():
            return True
        first_chars = [w[0].lower() for w in definition.split()
                       if w.lower() not in connecting_words]
        if "".join(first_chars) == abb.lower() or "".join(first_chars) == abb[:-1].lower():
            return True
        return False

    def get_char_overlap(str1, str2):
        a_counter = Counter("".join(clean_text(str1).split()))
        b_counter = Counter("".join(clean_text(str2).split()))
        c_counter = Counter(
            {key: b_counter.get(key, 0) - value for key, value in a_counter.items()})
        c_counter_sum = sum([abs(val) for val in c_counter.values()])
        return abs(c_counter_sum / sum(a_counter.values()))

    label_dataset = expand_labels(dataset)
    to_be_merged = defaultdict(list)
    while True:
        merged_titles = set()
        for cand in label_dataset:
            if len(label_dataset[cand]) > 1:
                titles = sorted(
                    label_dataset[cand], reverse=True, key=lambda x: len(dataset[x]))
                if titles[0] in merged_titles:
                    continue
                abbs = [re.sub(r"[^A-Za-z]", "", get_abbreviation(title)).lower()
                        for title in titles]
                definitions = [
                    re.sub(r"[^A-Za-z]", "", get_definition(title)).lower() for title in titles]
                for title, abb, definition in zip(titles[1:], abbs[1:], definitions[1:]):
                    if title in merged_titles:
                        continue
                    if abb == abbs[0]:
                        to_be_merged[titles[0]].append(title)
                        merged_titles.add(title)
                    elif (abb[0] == abbs[0][0] and abb[-1] == abbs[0][-1] and
                          get_char_overlap(definitions[0], definition) < 0.2):
                        to_be_merged[titles[0]].append(title)
                        merged_titles.add(title)
                    elif ((abb == f"{abbs[0]}s" or f"{abb}s" == abbs[0] or
                           abb == f"{abbs[0]}'s" or f"{abb}'s" == abbs[0]) and
                          get_char_overlap(definitions[0], definition) < 0.2):
                        to_be_merged[titles[0]].append(title)
                        merged_titles.add(title)
                    elif ((abb in abbs[0] or abbs[0] in abb) and
                          get_char_overlap(definitions[0], definition) < 0.2):
                        to_be_merged[titles[0]].append(title)
                        merged_titles.add(title)
                    elif ((abb in abbs[0] and definition in definitions[0]) or
                          (abbs[0] in abb and definitions[0] in definition)):
                        to_be_merged[titles[0]].append(title)
                        merged_titles.add(title)
                    elif not set(abb).difference(set(abbs[0])) and get_char_overlap(definitions[0], definition) < 0.2:
                        to_be_merged[titles[0]].append(title)
                        merged_titles.add(title)
        if not to_be_merged:
            break
        count = 0
        for k, v in to_be_merged.items():
            count += len(v)
        for k, titles in to_be_merged.items():
            for title in titles:
                try:
                    dataset[k].extend(
                        [c for c in dataset[title] if c not in dataset[k]])
                    del dataset[title]
                except KeyError:
                    continue
        label_dataset = expand_labels(dataset)
        to_be_merged = defaultdict(list)

    to_be_merged = defaultdict(set)
    to_be_removed = set()
    for cand in label_dataset:
        if len(label_dataset[cand]) > 1:
            titles = sorted(label_dataset[cand], key=lambda x: len(dataset[x]))
            abbs = [re.sub(r"[^A-Za-z]", "", get_abbreviation(title)).lower()
                    for title in titles]
            definitions = [
                re.sub(r"[^A-Za-z]", " ", get_definition(title)).lower() for title in titles]
            if any([is_regular_title(title) and (cand in clean_label(definition) or
                                                 get_char_overlap(definition, cand) < 0.1)
                    for title, definition in zip(titles, definitions)]):
                mod = False
                rem_titles = len(titles)
                for title, abb, definition in zip(titles, abbs, definitions):
                    if title not in dataset:
                        continue
                    if not is_regular_title(title) and rem_titles > 1:
                        dataset[title] = [label for label in dataset[title]
                                          if cand not in clean_label(label)]
                        if len(dataset[title]) == 0:
                            del dataset[title]
                        mod = True
                        rem_titles -= 1
                    elif (is_regular_title(title) and
                          (cand not in clean_label(definition) and get_char_overlap(definition, cand) > 0.1) and
                            rem_titles > 1):
                        dataset[title] = [label for label in dataset[title]
                                          if cand not in clean_label(label)]
                        if len(dataset[title]) == 0:
                            del dataset[title]
                        mod = True
                        rem_titles -= 1
                if mod:
                    label_dataset = expand_labels(dataset)

    label_dataset = expand_labels(dataset)
    while has_overlap(label_dataset):
        dataset = merge_dataset(dataset, label_dataset)
        label_dataset = expand_labels(dataset)

    # cleanup
    for k in dataset.copy():
        if len(dataset[k]) == 0:
            del dataset[k]

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    def construct_new_candidates(candidate1, candidate2, delimiter):
        candidates = []
        if "(" in candidate1:
            abb1 = candidate1.split("(")[0].strip()
            long1 = candidate1.split("(")[1][:-1].strip()
        else:
            abb1 = None
            long1 = None
        if "(" in candidate2:
            abb2 = candidate2.split("(")[0].strip()
            long2 = candidate2.split("(")[1][:-1].strip()
        else:
            abb2 = None
            long2 = None
        for sub1 in [abb1, long1, candidate1]:
            for sub2 in [abb2, long2, candidate2]:
                if sub1 and sub2:
                    candidates.append(f"{sub1}{delimiter}{sub2}")
        return candidates

    def find_title(label_dataset_mapping, title):
        if clean_label(title) in label_dataset_mapping:
            return label_dataset_mapping[clean_label(title)]["group_name"]

    for sentence_candidate in sentences_candidates:
        original_sentence = sentence_candidate["original_sentence"]
        n_masks = len(sentence_candidate["candidate_sentences"])
        # unmasking the masks that are excluded
        for i, candidate_sentence in enumerate(sentence_candidate["candidate_sentences"]):
            candidate = candidate_sentence["candidate"]
            if clean_label(candidate) not in label_dataset_mapping:
                sentence_candidate["masked_sentence"] = sentence_candidate["masked_sentence"].replace(
                    f"@CAND{i}#", candidate)
                sentence_candidate["candidate_sentences"][i]["candidate"] = None
        # Joining candidates
        masked_sentence = sentence_candidate["masked_sentence"]
        reg_delimiters = [" ", "\'s ", "\-", "\- "]
        for reg_delimiter in reg_delimiters:
            pattern = re.compile(rf"@CAND(\d)#{reg_delimiter}@CAND(\d)#")
            matches = reversed(list(pattern.finditer(masked_sentence)))
            if not matches:
                continue
            for match in matches:
                delimiter = reg_delimiter.replace("\\", "")
                i1, i2 = int(match.group(1)), int(match.group(2))
                candidate1 = sentence_candidate["candidate_sentences"][i1]["candidate"]
                candidate2 = sentence_candidate["candidate_sentences"][i2]["candidate"]
                if not candidate1 or not candidate2:
                    continue
                candidate2_abb = candidate2[candidate2.rfind("(")+1:].strip()
                if candidate2_abb.endswith(")"):
                    candidate2_abb = candidate2_abb[:-1]
                if len(candidate2_abb) < 3:
                    continue
                new_candidate = f"{candidate1}{delimiter}{candidate2}"
                title = find_title(label_dataset_mapping, candidate2)
                if not title:
                    title = new_candidate
                    dataset[title] = []
                candidate_1_2 = f"{candidate1.split('(')[0].strip()} {candidate2.split('(')[0].strip()}"
                clean_candidate_1_2 = clean_label(candidate_1_2)
                if clean_candidate_1_2 in label_dataset_mapping:
                    title = label_dataset_mapping[clean_candidate_1_2]["group_name"]
                new_candidates = construct_new_candidates(
                    f"{candidate1}", f"{candidate2}", delimiter=delimiter
                )
                dataset[title].extend(
                    [c for c in new_candidates if c not in dataset[title]]
                )
                sentence_candidate["candidate_sentences"][i2]["candidate"] = new_candidate
                sentence_candidate["candidate_sentences"][i2]["sentence"] = original_sentence.replace(
                    new_candidate, "@CAND#"
                )
                sentence_candidate["candidate_sentences"][i1]["candidate"] = None
                sentence_candidate["masked_sentence"] = sentence_candidate["masked_sentence"].replace(
                    match.group(0), f"@CAND{i2}#"
                )

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    label_dataset = expand_labels(dataset)
    while has_overlap(label_dataset):
        dataset = merge_dataset(dataset, label_dataset)
        label_dataset = expand_labels(dataset)

    # cleanup
    for k in dataset.copy():
        if len(dataset[k]) == 0:
            del dataset[k]

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    def to_merge(group1, group2, abb1, abb2):
        def overlap(term1, term2, abb1, abb2):
            term1 = term1.replace("'s", "")
            term2 = term2.replace("'s", "")
            connecting_words = ["of", "in", "on", "and", "for", "from",
                                "the", "in", "a", "to", "after", "with", "at", "by"]
            term1_words = set([w for w in term1.replace(
                abb1, "").split() if w not in connecting_words])
            term2_words = set([w for w in term2.replace(
                abb2, "").split() if w not in connecting_words])
            cleaned_term1_words = set([w for w in clean_label(
                term1).split() if w not in connecting_words])
            cleaned_term2_words = set([w for w in clean_label(
                term2).split() if w not in connecting_words])
            if clean_label(term1) in clean_label(term2) or clean_label(term2) in clean_label(term1):
                return True
            if len(cleaned_term1_words.intersection(cleaned_term2_words)) > max(len(term1_words), len(term2_words)) / 2:
                return True
            if cleaned_term1_words.intersection(cleaned_term2_words) and any([w for w in term1_words if w in abb]):
                return True
            if cleaned_term1_words.intersection(cleaned_term2_words) and any([w for w in term2_words if w in abb]):
                return True
        return any(overlap(term1, term2, abb1, abb2) for term1 in group1 for term2 in group2 if "(" not in term1 and "(" not in term2)

    # Further merging
    for title1 in dataset.copy():
        if title1 not in dataset:
            continue
        abb1 = title1.split("(")[-1][:-1]
        if len(abb1) < 4:
            continue
        for title2 in dataset.copy():
            if title2 not in dataset:
                continue
            abb2 = title2.split("(")[-1][:-1]
            if len(abb2) < 4 or abb1 == abb2:
                continue
            if abb1.lower() == abb2.lower():
                if to_merge(dataset[title1], dataset[title2], abb1, abb2):
                    dataset[title1].extend(
                        [c for c in dataset[title2] if c not in dataset[title1]]
                    )
                    del dataset[title2]
    # Further merging
    for title1 in dataset.copy():
        if title1 not in dataset:
            continue
        abb1 = title1.split("(")[1][:-1]
        if len(abb1) < 4:
            continue
        for title2 in dataset.copy():
            if title2 not in dataset:
                continue
            abb2 = title2.split("(")[1][:-1]
            if len(abb2) < 4 or abb1 == abb2:
                continue
            if abb2 == f"{abb1}s" and to_merge(dataset[title1], dataset[title2], abb1, abb2):
                dataset[title1].extend(
                    [c for c in dataset[title2] if c not in dataset[title1]]
                )
                del dataset[title2]

    label_dataset = expand_labels(dataset)
    while has_overlap(label_dataset):
        dataset = merge_dataset(dataset, label_dataset)
        label_dataset = expand_labels(dataset)

    # cleanup
    for k in dataset.copy():
        if len(dataset[k]) == 0:
            del dataset[k]

    # Remove duplicates
    for k, v in dataset.items():
        dataset[k] = list(set(v))

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    # add abbreviations
    most_common_abbs = set()
    for k, v in dataset.items():
        abbs_counter = Counter()
        for cand in v:
            if cand.count("(") == 1:
                abb = cand.split("(")[1][:-1]
                if ")" in abb:
                    continue
                abbs_counter[abb] += 1
        try:
            _, count = abbs_counter.most_common()[0]
            for most_common, cur_count in abbs_counter.most_common():
                if cur_count == count:
                    most_common_abbs.add(most_common)
                else:
                    break
        except IndexError:
            continue
    for k, v in dataset.items():
        abbs_counter = Counter()
        abbs = []
        most_commons = set()
        for cand in v:
            if cand.count("(") == 1:
                abb = cand.split("(")[1][:-1].strip()
                definition = cand.split("(")[0].strip()
                if ")" in abb:
                    continue
                if (abb[0].lower() == definition.split()[0][0].lower() and
                        abb[-1].lower() == definition.split()[-1][0].lower()):
                    if abb not in abbs:
                        abbs.append(abb)
                abbs_counter[abb] += 1
        try:
            _, count = abbs_counter.most_common()[0]
            for most_common, cur_count in abbs_counter.most_common():
                if cur_count == count:
                    most_commons.add(most_common)
                else:
                    break
        except IndexError:
            continue
        for abb in abbs_counter:
            if abb in most_common_abbs and abb not in most_commons:
                continue
            if abb not in abbs:
                abbs.append(abb)
        dataset[k].extend(abbs)

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    to_be_removed = set()
    for k in dataset:
        opening_parenthesis_index = k.rfind("(")
        abb = k[opening_parenthesis_index+1:][:-1].strip()
        if len(abb.split()) == 1:
            continue
        for k2 in dataset:
            if k == k2:
                continue
            opening_parenthesis_index = k2.rfind("(")
            abb2 = k2[opening_parenthesis_index+1:][:-1].strip()
            if abb2.lower() == [ab.lower() for ab in abb.split()][-1]:
                for cand in dataset[k]:
                    if len(cand.split()) == 1:
                        continue
                    if any([cand.endswith(cand2) for cand2 in dataset[k2] if len(cand2.split()) > 1 and len(cand2) > 9]):
                        dataset[k2].extend(
                            [c for c in dataset[k] if c not in dataset[k2]])
                        to_be_removed.add(k)
                        break

    for k in dataset.copy():
        if k in to_be_removed:
            del dataset[k]

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    to_be_removed = set()
    for k in dataset:
        opening_parenthesis_index = k.rfind("(")
        abb = k[opening_parenthesis_index+1:][:-1].strip()
        definition = k[:opening_parenthesis_index]
        if len(abb) < 4:
            continue
        for k2 in dataset:
            if k == k2:
                continue
            opening_parenthesis_index = k2.rfind("(")
            abb2 = k2[opening_parenthesis_index+1:][:-1].strip()
            long2 = k2[:opening_parenthesis_index]
            if len(abb2) < 3:
                continue
            if len(abb) > 2 * len(abb2) + 1 and len(abb.replace(abb2, "").split()) > 2:
                continue
            if abb.endswith(abb2) and definition.endswith(long2):
                if not k2 in to_be_removed and not k in to_be_removed:
                    dataset[k2].extend(
                        [c for c in dataset[k] if c not in dataset[k2]])
                    to_be_removed.add(k)
    for k in to_be_removed:
        if k in dataset:
            del dataset[k]

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    # Remove duplicates
    for k, v in dataset.items():
        dataset[k] = list(set(v))

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    for k in dataset.copy():
        abb = k[k.rfind("(")+1:-1]
        if upper_count(abb) < 2:
            del dataset[k]

    abbreviations = set()
    for k, v in dataset.items():
        for cand in v:
            if len(cand.split()) == 1 and cand != k[:k.rfind("(")].strip():
                abbreviations.add(cand)

    for k in dataset.copy():
        definition = k[:k.rfind("(")].strip()
        if k.split()[0] in abbreviations and not has_pascal_pattern(definition.replace(k.split()[0], "")):
            del dataset[k]

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    for k, v in dataset.items():
        abb = k[k.rfind("(")+1:-1].strip()
        if re.sub(r"[^A-Za-z]", "", abb)[-1].lower() == "m":
            continue
        if any([cand.lower().endswith("model") for cand in v]):
            abb = k[k.rfind("(")+1:-1].strip()
            dataset[k].append(f"{abb} model")

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    for k in dataset.copy():
        abb = k[k.rfind("(")+1:-1].strip()
        definition = k[:k.rfind("(")].strip()
        words = definition.split()
        if len(abb) < 3:
            if len(words) == 1 and len([c for c in words[0] if c == c.upper()]) < 2:
                del dataset[k]
            elif len(words) > 1:
                del dataset[k]
        elif len(definition.split()) == 2 and len(definition) < 15:
            del dataset[k]
        elif len(words) == 1 and len([c for c in words[0] if c == c.upper()]) < 2:
            del dataset[k]
        elif len(words) > 1 and len(definition) < 15:
            del dataset[k]

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    not_found = defaultdict(list)
    title_mapping = defaultdict(set)
    for k, v in title_labels.items():
        for cand in v:
            if cand.startswith("the ") or cand.startswith("The "):
                cand = cand[4:]
            if clean_label(cand) in label_dataset_mapping:
                title_mapping[k].add(
                    label_dataset_mapping[clean_label(cand)]["group_name"]
                )
            else:
                not_found[k].append(cand)

    for title in title_mapping:
        titles = sorted(
            list(title_mapping[title]), reverse=True, key=lambda x: len(dataset[x])
        )
        dataset[titles[0]].extend(
            [c for c in title_labels[title] if c not in dataset[titles[0]]]
        )

    # Remove duplicates
    for k, v in dataset.items():
        filtered = []
        for cand in v:
            if cand not in filtered:
                filtered.append(cand)
        dataset[k] = filtered

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    for title in not_found:
        if title not in title_mapping:
            dataset[title] = title_labels[title]

    # Remove duplicates
    for k, v in dataset.items():
        filtered = []
        for cand in v:
            if cand not in filtered:
                filtered.append(cand)
        dataset[k] = filtered

    log.info(f'Number of groups: {len(dataset)}')
    count = 0
    for k, v in dataset.items():
        count += len(v)
    log.info(f'Number of candidates: {count}')

    label_dataset_mapping = construct_label_dataset_mapping(dataset)

    return dataset
