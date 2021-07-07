import logging
import regex
import re
import numpy as np
from collections import defaultdict, Counter

"""
Implementation of Shwartz-Hearst algorithm, with some refinement from Hayk Sargsyan

Based on implementation from https://github.com/philgooch/abbreviation-extraction
Added some refinements that improves the extraction and some filtering of extracted
abbreviations

"""
log = logging.getLogger(__name__)


class Candidate(str):
    def __init__(self, value):
        super().__init__()
        self.start = 0
        self.stop = 0

    def set_position(self, start, stop):
        self.start = start
        self.stop = stop


def yield_sentences(sentences):
    for sentence in sentences:
        yield sentence.strip()


def best_candidates(sentence):
    """
    :param sentence: line read from input file
    :return: a Candidate iterator
    """
    # mask http. Interferes with abbreviation extraction
    sentence = sentence.replace("http", ";")

    if '(' in sentence:

        close_index = -1
        while 1:
            # Look for open parenthesis. Need leading whitespace to avoid matching mathematical and chemical formulae
            open_index = sentence.find(' (', close_index + 1)

            if open_index == -1:
                break

            # Advance beyond whitespace
            open_index += 1

            # Look for closing parentheses
            close_index = open_index + 1
            open_count = 1
            skip = False
            while open_count:
                try:
                    char = sentence[close_index]
                except IndexError:
                    # We found an opening bracket but no associated closing bracket
                    # Skip the opening bracket
                    skip = True
                    break

                if char == '(':
                    if open_count == 1:
                        close_index -= 2
                        skip = True
                        break
                    open_count += 1
                elif char in [')', ';', ':', ',', '%', '=', '.']:
                    open_count -= 1
                close_index += 1

            if skip:
                close_index = open_index + 1
                continue

            # Output if conditions are met
            start = open_index + 1
            stop = close_index - 1
            if sentence[start:stop].endswith("-"):
                stop -= 1
            candidate = sentence[start:stop]

            # Take into account whitespace that should be removed
            start = start + len(candidate) - len(candidate.lstrip())
            stop = stop - len(candidate) + len(candidate.rstrip())
            candidate = sentence[start:stop]
            if conditions(candidate):
                new_candidate = Candidate(candidate)
                new_candidate.set_position(start, stop)
                yield new_candidate


def conditions(candidate):
    """
    Based on Schwartz&Hearst

    2 <= len(str) <= 10
    len(tokens) <= 2
    re.search(r'\p{L}', str)
    str[0].isalnum()

    and extra:
    if it matches (\p{L}\.?\s?){2,}
    it is a good candidate.

    :param candidate: candidate abbreviation
    :return: True if this is a good candidate
    """
    if not candidate:
        return False
    viable = True
    if regex.match(r'(\p{L}\.?\s?){2,}', candidate.lstrip()):
        viable = True
    if len(candidate) < 2 or len(candidate) > 10:
        viable = False
    if len(candidate.split()) > 2:
        viable = False
    if not regex.search(r'\p{L}', candidate):
        viable = False
    if not candidate[0].isalnum():
        viable = False

    return viable


def get_definition(candidate, sentence):
    """
    Takes a candidate and a sentence and returns the definition candidate.

    The definition candidate is the set of tokens (in front of the candidate)
    that starts with a token starting with the first character of the candidate

    :param candidate: candidate abbreviation
    :param sentence: current sentence (single line from input file)
    :return: candidate definition for this abbreviation
    """
    # Take the tokens in front of the candidate
    tokens = regex.split(r'[\s\-]+', sentence[:candidate.start - 2].lower())
    # the char that we are looking for
    key = candidate[0].lower()

    # Special treatment for abbreviations starting with a letter of connecting words:
    # "a": "and"
    # "o": "of", "on", "or"
    # "f": "for", "from"
    # "i": "in"
    # "t": "to"
    # Replace the first chars of the connecting words by a special char "u", in order not to
    # count the connecting words: otherwise some definitions are cut-off at the connecting word
    if key == "a":
        first_chars = [t[0] if t != "and" else "ü" for t in tokens]
    elif key == "o":
        first_chars = [t[0] if t != "of" and t !=
                       "on" and t != "or" else "ü" for t in tokens]
    elif key == "f":
        first_chars = [t[0] if t != "for" and t !=
                       "from" else "ü" for t in tokens]
    elif key == "i":
        first_chars = [t[0] if t != "in" else "ü" for t in tokens]
    elif key == "t":
        first_chars = [t[0] if t != "to" else "ü" for t in tokens]
    else:
        first_chars = [t[0] for t in tokens]

    # Count the number of tokens that start with the same character as the candidate
    definition_freq = first_chars.count(key)
    candidate_freq = candidate.lower().count(key)

    # Look for the list of tokens in front of candidate that
    # have a sufficient number of tokens starting with key

    if candidate_freq <= definition_freq:
        # we should at least have a good number of starts
        count = 0
        start = 0
        start_index = len(first_chars) - 1
        while count < candidate_freq:
            if abs(start) > len(first_chars):
                raise ValueError("candidate {} not found".format(candidate))
            start -= 1
            # Look up key in the definition
            try:
                start_index = first_chars.index(key, len(first_chars) + start)
            except ValueError:
                pass

            # Count the number of keys in definition
            start_temp = len(' '.join(tokens[:start_index]))
            stop_temp = candidate.start - 1
            candidate_temp = sentence[start_temp:stop_temp]
            if len(candidate_temp) < len(candidate):
                continue
            count = first_chars[start_index:].count(key)

        # We found enough keys in the definition so return the definition as a definition candidate
        start = len(' '.join(tokens[:start_index]))
        stop = candidate.start - 1
        candidate = sentence[start:stop]

        # Remove whitespace
        start = start + len(candidate) - len(candidate.lstrip())
        stop = stop - len(candidate) + len(candidate.rstrip())
        candidate = sentence[start:stop]

        new_candidate = Candidate(candidate)
        new_candidate.set_position(start, stop)
        return new_candidate

    else:
        raise ValueError(
            'There are less keys in the tokens in front of candidate than there are in the candidate')


def select_definition(definition, abbrev):
    """
    Takes a definition candidate and an abbreviation candidate
    and returns True if the chars in the abbreviation occur in the definition

    Based on
    A simple algorithm for identifying abbreviation definitions in biomedical texts, Schwartz & Hearst
    :param definition: candidate definition
    :param abbrev: candidate abbreviation
    :return:
    """

    if len(definition) < len(abbrev):
        raise ValueError('Abbreviation is longer than definition')

    if abbrev in definition.split():
        raise ValueError('Abbreviation is full word of definition')

    if abbrev in [w.strip() for w in re.sub('[^A-Za-z0-9&]+', ' ', definition).split()]:
        raise ValueError('Abbreviation is full word of definition')

    s_index = -1
    l_index = -1

    while 1:
        try:
            long_char = definition[l_index].lower()
        except IndexError:
            raise

        short_char = abbrev[s_index].lower()

        if not short_char.isalnum():
            s_index -= 1

        if s_index == -1 * len(abbrev):
            if short_char == long_char:
                if l_index == -1 * len(definition) or not definition[l_index - 1].isalnum():
                    break
                else:
                    l_index -= 1
            else:
                l_index -= 1
                if l_index == -1 * (len(definition) + 1):
                    raise ValueError(
                        "definition {} was not found in {}".format(abbrev, definition))

        else:
            if short_char == long_char:
                s_index -= 1
                l_index -= 1
            else:
                l_index -= 1

    new_candidate = Candidate(definition[l_index:len(definition)])
    new_candidate.set_position(definition.start, definition.stop)
    first_token = new_candidate.split()[0].strip()
    if (not first_token == "and" and not first_token == "or" and
            not first_token == "of" and not first_token == "from" and
            not first_token == "on" and not first_token == "for" and
            not first_token == "in" and not first_token == "to"):
        definition = new_candidate

    tokens = len(definition.split())
    length = len(abbrev)

    if tokens > min([length + 5, length * 2]):
        raise ValueError("did not meet min(|A|+5, |A|*2) constraint")

    # Do not return definitions that contain unbalanced parentheses
    if definition.count('(') != definition.count(')'):
        raise ValueError("Unbalanced parentheses not allowed in a definition")

    return definition


def clean_candidate(candidate):
    cleaned_candidate = candidate
    try:
        cleaned_candidate = re.sub(
            r"[0-9\-\:\/]", " ", candidate).split()[0].strip()
    except IndexError:
        return candidate
    return cleaned_candidate


def select_definition_new(definition, abbrev, repeated=False, old=None):
    """
    Takes a definition candidate and an abbreviation candidate
    and returns True if the chars in the abbreviation occur in the definition

    Same as select_definition, but applies some cleaning on the definition first
    :param definition: candidate definition
    :param abbrev: candidate abbreviation
    :return:
    """

    if len(definition) < len(abbrev):
        raise ValueError('Abbreviation is longer than definition')

    s_index = -1
    l_index = -1

    while 1:
        try:
            long_char = definition[l_index]
            if l_index == -1 * len(definition) or not definition[l_index-1].isalnum():
                long_char = long_char.upper()
        except IndexError:
            cleaned_abbrev = clean_candidate(abbrev)
            if cleaned_abbrev != abbrev:
                return select_definition_new(definition, cleaned_abbrev, repeated=True, old=abbrev)
            else:
                raise

        short_char = abbrev[s_index]

        if not short_char.isalnum():
            s_index -= 1

        if s_index == -1 * len(abbrev):
            if short_char == long_char:
                if l_index == -1 * len(definition) or not definition[l_index - 1].isalnum():
                    break
                else:
                    l_index -= 1
            else:
                l_index -= 1
                if l_index == -1 * (len(definition) + 1):
                    raise ValueError(
                        "definition {} was not found in {}".format(abbrev, definition))

        else:
            if short_char == long_char:
                s_index -= 1
                l_index -= 1
            else:
                l_index -= 1

    new_candidate = Candidate(definition[l_index:len(definition)])
    new_candidate.set_position(definition.start, definition.stop)
    first_token = new_candidate.split()[0].strip()
    if (not first_token == "and" and not first_token == "or" and
            not first_token == "of" and not first_token == "from" and
            not first_token == "on" and not first_token == "for" and
            not first_token == "in" and not first_token == "to"):
        definition = new_candidate

    if abbrev in [w.strip() for w in re.sub('[^A-Za-z0-9&]+', ' ', definition).split()]:
        raise ValueError('Abbreviation is full word of definition')

    tokens = len(definition.split())
    length = len(abbrev)

    if tokens > min([length + 5, length * 2]):
        raise ValueError("did not meet min(|A|+5, |A|*2) constraint")

    # Do not return definitions that contain unbalanced parentheses
    if definition.count('(') != definition.count(')'):
        raise ValueError("Unbalanced parentheses not allowed in a definition")

    if repeated == True:
        if len(abbrev) < 2 or abbrev != abbrev.upper():
            raise ValueError("Unreliable subpart from the original candidate")

    return definition


def mask(sentence, term, start, end):
    return sentence[:start] + term + sentence[end:]


def unmask(sentence, masks):
    for mask in masks:
        sentence = sentence.replace(mask[0], mask[1])
    return sentence


def construct_sentence_obj(sentence, current_candidate_definitions, Id):
    """Constructs masked sentence objects for later use"""
    ret = {
        "original_sentence": sentence,
        "Id": Id,
        "masked_sentence": sentence,
        "candidate_sentences": [],
        "masks": []
    }
    i = 0
    for candidate, definition in current_candidate_definitions:
        # definition (candidate)
        start = ret["masked_sentence"].find(f"{definition} ({candidate})")
        while start != -1:
            end = len(f"{definition} ({candidate})") + start
            ret["masked_sentence"] = mask(
                ret["masked_sentence"], f"@CAND{i}#", start, end)
            ret["candidate_sentences"].append({
                "sentence": unmask(ret["masked_sentence"], ret["masks"]).replace(
                    f"@CAND{i}#", "@CAND#"),
                "candidate": f"{definition} ({candidate})"
            })
            ret["masks"].append([f"@CAND{i}#", f"{definition} ({candidate})"])
            i += 1
            start = ret["masked_sentence"].find(f"{definition} ({candidate})")
        # only definition
        start = ret["masked_sentence"].find(f"{definition}")
        if definition in "@CAND#":
            continue  # make sure there are no endless loops
        while start != -1:
            if start > 0 and re.search(r"[A-Za-z0-9]", ret["masked_sentence"][start-1]):
                break
            end = len(f"{definition}") + start
            if (end < len(ret["masked_sentence"]) and
                    re.search(r"[A-Za-z\)\]]", ret["masked_sentence"][end])):
                break
            ret["masked_sentence"] = mask(
                ret["masked_sentence"], f"@CAND{i}#", start, end)
            ret["candidate_sentences"].append({
                "sentence": unmask(ret["masked_sentence"], ret["masks"]).replace(
                    f"@CAND{i}#", "@CAND#"),
                "candidate": f"{definition}"
            })
            ret["masks"].append([f"@CAND{i}#", f"{definition}"])
            i += 1
            start = ret["masked_sentence"].find(f"{definition}")

        # only candidate
        start = ret["masked_sentence"].find(f"{candidate}")
        if candidate in "@CAND#":
            continue  # make sure there are no endless loops
        while start != -1:
            if start > 0 and re.search(r"[A-Za-z0-9\,\(\[\;\.]", ret["masked_sentence"][start-1]):
                break
            end = len(f"{candidate}") + start
            if (end < len(ret["masked_sentence"]) and
                    re.search(r"[A-Za-z\)\]]", ret["masked_sentence"][end])):
                break
            ret["masked_sentence"] = mask(
                ret["masked_sentence"], f"@CAND{i}#", start, end)
            ret["candidate_sentences"].append({
                "sentence": unmask(ret["masked_sentence"], ret["masks"]).replace(
                    f"@CAND{i}#", "@CAND#"),
                "candidate": f"{candidate}"
            })
            ret["masks"].append([f"@CAND{i}#", f"{candidate}"])
            i += 1
            start = ret["masked_sentence"].find(f"{candidate}")
    return ret


def reject_definition(definition, abbrev):
    """Rejects definitions based on some simple patterns"""
    # Filter out too short definitions
    if len(definition) < 6:
        return True

    # Filter out too long definitions
    if len(definition.split()) > 2 * len(abbrev):
        return True

    # Filter out definitions that are references, identified by "et al"
    if " et al " in definition:
        return True

    # Filter out definitions that are clearly generic sentences
    if " were " in definition or " are " in definition or " is " in definition:
        return True
    if " taken from " in definition:
        return True
    if (definition.startswith("were ") or definition.startswith("are ") or
            definition.startswith("is ") or definition.startswith("as ")):
        return True
    if "i.e." in definition or "e.g." in definition or "which" in definition:
        return True
    if (definition.lower().startswith("using ") or
        definition.lower().startswith("use ") or
        definition.lower().startswith("used ") or
            " using " in definition.lower() or
            " used " in definition.lower()):
        return True

    # Filter out definitions with special symbols
    exclusion_symbols = "[]()%=}{$"
    for exclusion_symbol in exclusion_symbols:
        if exclusion_symbol in definition:
            return True
    return False


def reject_abbreviation(abbrev, existing_abbreviations):
    """Rejects abbreviations based on some simple patterns"""
    # Filter out figures and tables
    figure_tables = ["fig.", "fig ", " fig", "Fig.", "Fig ", " Fig", "figur", "Figur",
                     "tab.", "tab ", " tab", "Tab.", "Tab ", " Tab", "table", "Table"]
    if any([fig_tab in abbrev for fig_tab in figure_tables]):
        return True

    # Filter out candidates that start with connecting words
    if abbrev.startswith("or ") or abbrev.startswith("and "):
        return True
    if (abbrev.startswith("for ") or abbrev.startswith("from ") or
            abbrev.startswith("with ")):
        return True

    # Filter out candidates with special symbols
    exclusion_symbols = "[]()%=,$"
    for exclusion_symbol in exclusion_symbols:
        if exclusion_symbol in abbrev:
            return True

    # Filter out candidates with spaces and short pieces
    abbrev_words = abbrev.split()
    if len(abbrev_words) > 1 and min([len(w) for w in abbrev_words]) == 1:
        longest_word = abbrev_words[np.argmax([len(w) for w in abbrev_words])]
        if ((len(longest_word) < 3 or
             regex.search(r"[^A-Z]", longest_word))):
            return True

    # Filter out all lowercase multi-word abbreviations
    if len(abbrev_words) > 1 and abbrev == abbrev.lower():
        return True

    # Filter out abbreviations that have less then 2 upper cases unless in existing_abbreviations
    if (len([c for c in abbrev if regex.search(r"[A-Z]", c)]) < 2 and
            abbrev.lower() not in existing_abbreviations):
        return True

    # Filter out some common words and short non-uppercase abbreviations
    common_words = ["see", "See", "fig", "Fig", "figs", "Figs", "total", "Total", "ie",
                    "eg", "eq", "Eq", "eqs", "Eqs", "vol", "Vol"]
    if abbrev in common_words or (abbrev != abbrev.upper() and len(abbrev) < 3):
        return True

    # Filter out reference-like mentions
    if abbrev.lower().startswith("in ") or abbrev.lower().startswith("see "):
        return True
    if "i.e." in abbrev or "e.g." in abbrev:
        return True

    return False


def extract_abbreviation_definition_pairs(sentences, ids, existing_abbreviations=set()):
    abbrev_map = dict()
    list_abbrev_map = defaultdict(list)
    counter_abbrev_map = dict()
    omit = 0
    written = 0
    sentence_iterator = enumerate(yield_sentences(sentences))

    sentences_candidates = []
    ids_abbreviations = {}
    for i, sentence in sentence_iterator:
        if ids[i] not in ids_abbreviations:
            ids_abbreviations[ids[i]] = {}
        current_candidate_definitions = []
        # Remove any quotes around potential candidate terms
        clean_sentence = regex.sub(
            r'([(])[\'"\p{Pi}]|[\'"\p{Pf}]([);:])', r'\1\2', sentence)
        try:
            for candidate in best_candidates(clean_sentence):
                try:
                    definition = get_definition(candidate, clean_sentence)
                except (ValueError, IndexError) as e:
                    omit += 1
                else:
                    try:
                        definition = select_definition_new(
                            definition, candidate)
                        if (reject_definition(definition, candidate) or
                                reject_abbreviation(candidate, existing_abbreviations)):
                            definition = select_definition(
                                definition, candidate)
                    except (ValueError, IndexError) as e:
                        try:
                            definition = select_definition(
                                definition, candidate)
                        except (ValueError, IndexError) as e:
                            omit += 1
                            definition = None
                    if definition is not None:
                        # filter out very short definitions
                        if (reject_definition(definition, candidate) or
                                reject_abbreviation(candidate, existing_abbreviations)):
                            continue
                        # Either append the current definition to the list of previous definitions ...
                        current_candidate_definitions.append(
                            (candidate, definition))
                        list_abbrev_map[candidate].append(definition)
                        written += 1
        except (ValueError, IndexError) as e:
            pass

        sentences_candidates.append(construct_sentence_obj(
            sentence, current_candidate_definitions, ids[i]))
        for candidate, definition in current_candidate_definitions:
            if candidate not in ids_abbreviations[ids[i]]:
                ids_abbreviations[ids[i]][candidate] = []
            if definition not in ids_abbreviations[ids[i]][candidate]:
                ids_abbreviations[ids[i]][candidate].append(definition)
    log.info(
        f'{written} abbreviations detected and kept ({omit} omitted)')

    # Retutn all definitions for each term
    for k, v in list_abbrev_map.items():
        counter_abbrev_map[k] = Counter(v)
    return {
        "abbreviations": counter_abbrev_map,
        "sentences_candidates": sentences_candidates,
        "ids_abbreviations": ids_abbreviations
    }
