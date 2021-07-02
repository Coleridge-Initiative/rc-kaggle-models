import math
import re
import json

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


def clean_text(txt):
    return re.sub("[^A-Za-z0-9]+", " ", str(txt).lower()).strip()


def check_number(string):
    string = clean_text(string)
    for w in string.split():
        if w.isdigit():
            return True
    return False


settings = json.load(open("./settings.json", "rb"))


ACCEPTED_AUGMENT_LABELS = []

AUGMENT_LABELS = pd.read_csv(
    f"./{settings['RAW_DATA_DIR']}/hard_labels.csv"
).label.tolist()


for i in range(len(AUGMENT_LABELS)):
    if (
        AUGMENT_LABELS[i].islower()
        or AUGMENT_LABELS[i][0].islower()
        or check_number(AUGMENT_LABELS[i][0])
    ):
        continue

    if (
        len(clean_text(AUGMENT_LABELS[i]).split()) >= 3
        and len(clean_text(AUGMENT_LABELS[i]).split()) <= 10
    ):
        ACCEPTED_AUGMENT_LABELS.append(AUGMENT_LABELS[i])

AUGMENT_LABELS = ACCEPTED_AUGMENT_LABELS
print(f"There are {len(AUGMENT_LABELS)} augment labels.")


class QueryDataLoader(Sequence):
    """Query Data Sampling Class."""

    def __init__(self, data, batch_size=32):
        self.batch_size = batch_size
        self.data = data.fillna("")
        self.batch_ids = self.data["id"].tolist()
        self.batch_text = self.data["text"].tolist()
        self.batch_label = self.data["label"].tolist()

    def __len__(self):
        return math.ceil(len(self.batch_text) / self.batch_size)

    def __getitem__(self, index):
        id = self.batch_ids[index * self.batch_size : (index + 1) * self.batch_size]
        text = self.batch_text[index * self.batch_size : (index + 1) * self.batch_size]
        label = self.batch_label[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        classes = [1 if l != "" else 0 for l in label]
        return id, text, label, classes


class PositiveQueryDataLoader(Sequence):
    """Positive Query Data Sampling Class."""

    def __init__(self, data, batch_size=32):
        self.batch_size = batch_size
        self.data = data.fillna("")
        self.data = self.data[self.data.label != ""]
        self.batch_ids = self.data["id"].tolist()
        self.batch_text = self.data["text"].tolist()
        self.batch_label = self.data["label"].tolist()

    def __len__(self):
        return math.ceil(len(self.batch_text) / self.batch_size)

    def __getitem__(self, index):
        id = self.batch_ids[index * self.batch_size : (index + 1) * self.batch_size]
        text = self.batch_text[index * self.batch_size : (index + 1) * self.batch_size]
        label = self.batch_label[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        classes = [1 if l != "" else 0 for l in label]
        return id, text, label, classes


class SupportQueryDataLoader(Sequence):
    """Support/Query Data Sampling for training models.

    Args:
        data: Pandas csv dataset
        tokenizer: A huggingface tokenizer
        training_steps: A number of training steps per epoch
        batch_size: A batch size
        is_train: True if it's a training stage else False
        K: Number of support samples per one query sample
        support_data_group: Group data of all positive training group
        query_dataloader: A Query Data loader for the inference/evaluation
        support_masked: True if we want to replace all dataset title in support samples by <MASK> token
        query_masked: True if we want to replace all dataset title in query samples by <MASK> token
        query_positive_random_prob: The probability of random query sample is Positive
        return_query_ids: True if you want return query sample ids
        return_query_labels: True if you want return query sample label for evaluation
        no_overlap_support_query_group: True if you want Support groups and Query groups are non-overlap
        allow_support_group_duplicate: True if you allow Support groups are duplicate in each batch
        donot_process_support: True if you don't want use Support samples, support batch will return empty
        norm_fn: Normalize function apply for input string and the dataset label.
    """

    def __init__(
        self,
        data,
        tokenizer,
        training_steps=500,
        batch_size=32,
        is_train=False,
        K=3,
        support_data_group=None,
        query_dataloader=None,
        support_masked=False,
        query_masked=False,
        query_positive_random_prob=0.0,
        return_query_ids=False,
        return_query_labels=False,
        no_overlap_support_query_group=True,
        allow_support_group_duplicate=False,
        donot_process_support=False,
        norm_fn=None,
    ):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data = data.fillna("")
        self.is_train = is_train
        self.K = K
        self.data_group = {}
        self._create_group_data()
        self.support_data_group = support_data_group
        self.len = training_steps
        self.query_dataloader = query_dataloader
        self.support_masked = support_masked
        self.query_masked = query_masked
        self.query_positive_random_prob = query_positive_random_prob
        self.return_query_ids = return_query_ids
        self.return_query_labels = return_query_labels
        self.no_overlap_support_query_group = no_overlap_support_query_group
        self.allow_support_group_duplicate = allow_support_group_duplicate
        self.donot_process_support = donot_process_support
        self.norm_fn = norm_fn

        self.on_epoch_end()

        all_unique_support_group = self.all_unique_group
        all_unique_query_group = self.all_unique_group

        self.all_unique_support_group = shuffle(
            list(set(all_unique_support_group) - set([0]))
        )
        self.all_unique_query_group = shuffle(
            list(set(all_unique_query_group) - set([0]))
        )

    def _augment_word_drop(self, original_string, label_string):
        """Random one label in label_string and drop its acronym if exists.

        Args:
            orignal_string: Original input string
            label_string: List of dataset label in the original string, separated by pipe charactor

        Returns:
            new_original_string and new_label_string

        Eg:
            Inputs:
                original_string: we use data from the National Study of Youth (NSY) and NELS:88 data...
                label_string: National Study of Youth (NSY)|NELS:88
            Outputs:
                new_original_string: we use data from the National Study of Youth and NELS:88 data...
                new_label_string: National Study of Youth|NELS:88
        """

        single_labels = label_string.split("|")
        idx_augmentable = []
        for i, l in enumerate(single_labels):
            if len(l.split(" ")) >= 4:
                idx_augmentable.append(i)

        if len(idx_augmentable) == 0:
            return original_string, label_string

        augment_idx = np.random.choice(idx_augmentable, size=1)[0]
        label = single_labels[augment_idx]

        label_words = label.split(" ")
        if "(" in label_words[-1].lower():
            # drop last words Acronym
            label_words = label_words[:-1]

        new_label = " ".join(label_words)
        single_labels[augment_idx] = new_label
        new_label_string = "|".join(single_labels)
        start_replace_idx = original_string.find(label)
        new_original_string = (
            original_string[:start_replace_idx]
            + new_label
            + original_string[start_replace_idx + len(label) :]
        )
        return new_original_string, new_label_string

    def _augment_lower(self, original_string, label_string):
        """Random one label in label_string and lower it except its acronym if exists.
        If acronym doesn't exist, the label will capitalize.

        Args:
            orignal_string: Original input string
            label_string: List of dataset label in the original string, separated by pipe charactor

        Returns:
            new_original_string and new_label_string

        Eg:
            Inputs:
                original_string: we use data from the National Study of Youth (NSY) and NELS:88 data...
                label_string: National Study of Youth (NSY)|NELS:88
            Outputs:
                new_original_string: we use data from the national study of youth (NSY) and NELS:88 data...
                new_label_string: national study of youth (NSY)|NELS:88
        """

        single_labels = label_string.split("|")
        idx_augmentable = []
        for i, l in enumerate(single_labels):
            if len(l.split(" ")) >= 3:
                idx_augmentable.append(i)

        if len(idx_augmentable) == 0:
            return original_string, label_string

        augment_idx = np.random.choice(idx_augmentable, size=1)[0]
        label = single_labels[augment_idx]
        new_label_words = label.split(" ")
        new_label_words = [
            w.lower() if "(" not in w else w for w in new_label_words
        ]  # do not lower the acronym
        new_label = " ".join(new_label_words)
        if "(" not in new_label:  # no acronym:
            new_label = new_label.capitalize()

        single_labels[augment_idx] = new_label
        new_label_string = "|".join(single_labels)
        start_replace_idx = original_string.find(label)
        new_original_string = (
            original_string[:start_replace_idx]
            + new_label
            + original_string[start_replace_idx + len(label) :]
        )
        return new_original_string, new_label_string

    def _augment_new_label(self, original_string, label_string):
        """Random one label in label_string and replace it by one of the labels from AUGMENT_LABELS.

        Args:
            orignal_string: Original input string
            label_string: List of dataset label in the original string, separated by pipe charactor

        Returns:
            new_original_string and new_label_string

        Eg:
            Inputs:
                original_string: we use data from the National Study of Youth (NSY) and NELS:88 data...
                label_string: National Study of Youth (NSY)|NELS:88
            Outputs:
                new_original_string: we use data from the National Assessment of Adult Literacy and NELS:88 data...
                new_label_string: National Assessment of Adult Literacy|NELS:88
        """

        single_labels = label_string.split("|")
        idx_augmentable = []
        for i, l in enumerate(single_labels):
            if len(l.split(" ")) >= 3:
                idx_augmentable.append(i)

        if len(idx_augmentable) == 0:
            return original_string, label_string

        augment_idx = np.random.choice(idx_augmentable, size=1)[0]
        label = single_labels[augment_idx]
        new_label = np.random.choice(AUGMENT_LABELS, size=1)[0]
        single_labels[augment_idx] = new_label
        new_label_string = "|".join(single_labels)
        start_replace_idx = original_string.find(label)
        new_original_string = (
            original_string[:start_replace_idx]
            + new_label
            + original_string[start_replace_idx + len(label) :]
        )
        return new_original_string, new_label_string

    def _create_group_data(self):
        """Create dictionary data group for training.

        Below are some notes about data group:
            1. Each group key is a unique dataset label idx and value is a list of title
               text, text_id and label.
            2. Group idx 0 is a negative group contains only negative samples (label is "").
            3. We only use 1/8 number of samples in Group 0 for training each single model without seed.
        """

        all_unique_group = list(self.data.group.unique())
        self.all_unique_group = all_unique_group
        for group in all_unique_group:
            self.data_group[group] = list(
                zip(
                    list(self.data[self.data["group"] == group].title),
                    list(self.data[self.data["group"] == group].text),
                    list(self.data[self.data["group"] == group].text_id),
                    list(self.data[self.data["group"] == group].label),
                )
            )

        # only get random 1/8 group 0 (negative samples)
        self.data_group[0] = shuffle(self.data_group[0])
        choosen_idxs = np.random.choice(
            list(range(0, len(self.data_group[0]))),
            size=len(self.data_group[0]) // 8,
            replace=False,
        )
        choosen_data_group_0 = []
        for c_i in choosen_idxs:
            choosen_data_group_0.append(self.data_group[0][c_i])
        self.data_group[0] = choosen_data_group_0
        print(f"Choose {len(self.data_group[0])} negative samples for training.")

    def on_epoch_end(self):
        """Shuffle data samples in each group after finish the epoch."""

        if self.is_train:
            for k in list(self.data_group.keys()):
                self.data_group[k] = shuffle(self.data_group[k])

    def __len__(self):
        return self.len

    def _gen_support_query_group(self, all_unique_support_group=None):
        """Random K support groups and 1 query group."""

        support_group = np.random.choice(
            self.all_unique_support_group
            if all_unique_support_group is None
            else list(set(all_unique_support_group) - set([0])),
            size=self.K,
            replace=self.allow_support_group_duplicate,
        )
        if self.no_overlap_support_query_group:
            query_group = np.random.choice(
                list(set(self.all_unique_query_group) - set(support_group)),
                size=1,
            )[0]
        else:
            query_group = np.random.choice(
                self.all_unique_query_group,
                size=1,
            )[0]
        return list(support_group), query_group

    def _take_one_positive_sample_in_group(self, group, data_group=None):
        """Random take one positive sample in the group

        Args:
            group: A group idx which the sample belongs
            data_group: A data group

        Returns:
            A tuples of (text_id, text, label)
        """

        assert group != 0, "Group must be != 0"
        if data_group is not None:
            group_data = data_group[group]
        else:
            group_data = self.data_group[group]
        choice_idx = np.random.randint(0, len(group_data), size=1)[0]
        return (
            group_data[choice_idx][2],  # text_id
            group_data[choice_idx][1],  # text
            group_data[choice_idx][-1],  # label
        )

    def _take_one_negative_sample_in_group(self, group=0, data_group=None):
        """Random take one negative sample in the group

        Args:
            group: A group idx which the sample belongs (always set to 0)
            data_group: A data group

        Returns:
            A tuples of (text_id, text, label)
        """

        group = 0  # 0 always negative samples.
        if data_group is not None:
            group_data = data_group[group]
        else:
            group_data = self.data_group[group]
        choice_idx = np.random.randint(0, len(group_data), size=1)[0]
        return (group_data[choice_idx][2], group_data[choice_idx][1], "")

    def __getitem__(self, index):
        # step 1: random choice K unique group for support set
        # and 1 unique group for query
        # note that we do not include group 0 in support set to
        # ensure that support set is only include positive samples.
        support_groups = []
        query_groups = []
        for _ in range(self.batch_size):
            support_group, query_group = self._gen_support_query_group(
                all_unique_support_group=list(self.support_data_group.keys())
                if self.support_data_group is not None
                else None
            )
            support_groups.append(support_group)
            query_groups.append(query_group)

        # step 2: create support/group data samples
        support_samples = []
        support_labels = []
        support_classes = []
        support_ids = []
        query_samples = []
        query_labels = []
        query_classes = []
        query_ids = []

        if self.query_dataloader is None:
            for query_group in query_groups:
                if np.random.random() <= self.query_positive_random_prob:
                    (
                        query_id,
                        query_text,
                        query_label,
                    ) = self._take_one_positive_sample_in_group(
                        query_group,
                    )
                else:
                    (
                        query_id,
                        query_text,
                        query_label,
                    ) = self._take_one_negative_sample_in_group(
                        query_group,
                    )
                query_ids.append(query_id)
                query_samples.append(query_text)
                query_labels.append(query_label)
                query_classes.append(1 if query_label != "" else 0)
                if self.return_query_ids is False:
                    query_ids = None
        else:
            (
                query_ids,
                query_samples,
                query_labels,
                query_classes,
            ) = self.query_dataloader.__getitem__(index)
            if self.return_query_ids is False:
                query_ids = None

        if self.donot_process_support is False:
            for support_group in support_groups:
                for group in support_group:
                    (
                        support_id,
                        support_text,
                        support_label,
                    ) = self._take_one_positive_sample_in_group(
                        group, data_group=self.support_data_group
                    )
                    support_ids.append(support_id)
                    support_samples.append(support_text)
                    support_labels.append(support_label)
                    support_classes.append(1)

        # step 3: tokenize and return computed sequence label
        support_batch = {}
        support_batch["input_ids"] = []
        support_batch["attention_mask"] = []
        support_batch["token_type_ids"] = []
        support_batch["sequence_labels"] = []
        support_batch["classes"] = []
        support_batch["ids"] = []

        query_batch = {}
        query_batch["input_ids"] = []
        query_batch["attention_mask"] = []
        query_batch["token_type_ids"] = []
        query_batch["sequence_labels"] = []
        query_batch["classes"] = []
        query_batch["ids"] = []
        query_batch["labels"] = []

        for i in range(len(query_samples)):
            out = self._process_data(
                query_samples[i], query_labels[i], masked_label=self.query_masked
            )
            query_batch["input_ids"].append(out["input_ids"])
            query_batch["attention_mask"].append(out["attention_mask"])
            query_batch["token_type_ids"].append(out["token_type_ids"])
            query_batch["sequence_labels"].append(out["sequence_labels"])
            query_batch["classes"].append(query_classes[i])
            if query_ids is not None:
                query_batch["ids"].append(query_ids[i])
            if self.return_query_labels:
                query_batch["labels"].append(query_labels[i])

        if self.donot_process_support is False:
            for i in range(len(support_samples)):
                out = self._process_data(
                    support_samples[i],
                    support_labels[i],
                    masked_label=self.support_masked,
                )
                support_batch["ids"].append(support_ids[i])
                support_batch["input_ids"].append(out["input_ids"])
                support_batch["attention_mask"].append(out["attention_mask"])
                support_batch["token_type_ids"].append(out["token_type_ids"])
                support_batch["sequence_labels"].append(out["sequence_labels"])
                support_batch["classes"].append(support_classes[i])

        # step 4: padding to max len
        query_batch["input_ids"] = pad_sequences(
            query_batch["input_ids"],
            padding="post",
            value=self.tokenizer.pad_token_id,
        )
        for k in ["attention_mask", "token_type_ids", "sequence_labels"]:
            pad_value = -100 if k == "sequence_labels" else 0
            query_batch[k] = pad_sequences(
                query_batch[k], padding="post", value=pad_value
            )

        if self.donot_process_support is False:
            support_batch["input_ids"] = pad_sequences(
                support_batch["input_ids"],
                padding="post",
                value=self.tokenizer.pad_token_id,
            )
            for k in ["attention_mask", "token_type_ids", "sequence_labels"]:
                pad_value = -100 if k == "sequence_labels" else 0
                support_batch[k] = pad_sequences(
                    support_batch[k], padding="post", value=pad_value
                )

        for k in list(support_batch.keys()):
            try:
                query_batch[k] = np.array(query_batch[k]).astype(np.int32)
                if self.donot_process_support is False:
                    support_batch[k] = np.array(support_batch[k]).astype(np.int32)
            except:
                pass

        return support_batch, query_batch

    def _process_data(self, inp_string, label_string, masked_label=False):
        """Preprocessing, augmentation for input and label string.

        Args:
            inp_string: raw input_string
            label_string: raw label-string
            masked_label: True if you want replace original label by <MASK>

        Returns:
            results: Dictionary of input_ids, attention_mask,
                token_type_ids and sequence_labels (for ner training)

        """
        inp_string = re.sub("\.+", " ", inp_string)
        inp_string = re.sub("\_+", " ", inp_string)
        inp_string = re.sub("\ +", " ", inp_string).strip()
        label_string = label_string.strip()
        if self.norm_fn is not None:
            inp_string = self.norm_fn(inp_string)
            label_string = self.norm_fn(label_string)
        if masked_label:
            for l_string in label_string.split("|"):
                inp_string = inp_string.replace(l_string, self.tokenizer.mask_token)
            label_string = self.tokenizer.mask_token

        input_tokenize = self.tokenizer(
            inp_string, return_offsets_mapping=True, max_length=320, truncation=True
        )
        input_offsets = input_tokenize["offset_mapping"][1:-1]
        sequence_labels = np.array([0] * len(input_tokenize["input_ids"]))

        # Augment here
        if self.is_train:
            if np.random.random() < 0.5:
                inp_string, label_string = self._augment_new_label(
                    inp_string, label_string
                )

            if np.random.random() < 0.5:
                if np.random.random() < 0.5:
                    inp_string, label_string = self._augment_lower(
                        inp_string, label_string
                    )
                else:
                    inp_string, label_string = self._augment_word_drop(
                        inp_string, label_string
                    )

        for l_string in label_string.split("|"):
            idx0, idx1 = self._find_match_position(inp_string, l_string)
            char_targets = [0] * len(inp_string)
            if idx0 != None and idx1 != None:
                if idx0 != None and idx1 != None:
                    for ct in range(idx0, idx1):
                        char_targets[ct] = 1

                target_idx = []
                for j, (offset1, offset2) in enumerate(input_offsets):
                    if sum(char_targets[offset1:offset2]) > 0:
                        target_idx.append(j)
                try:
                    sequence_labels[target_idx[0] + 1 : target_idx[-1] + 1 + 1] = 1
                except Exception as e:
                    pass

        results = {
            "input_ids": input_tokenize["input_ids"],
            "attention_mask": input_tokenize["attention_mask"],
            "token_type_ids": [0] * len(input_tokenize["input_ids"]),
            "sequence_labels": sequence_labels,
        }
        return results

    def _find_match_position(self, inp_string, label_string):
        """Find start end character index of label_string in inp_string

        Args:
            inp_string: raw input string
            label-string: raw label_string

        Returns:
            idx0: The start character index of label_string in inp_string
            idx1: The end character index of label_string in inp_string
        """

        inp_string = " ".join(str(inp_string).split()).strip()
        label_string = " ".join(str(label_string).split()).strip()

        if len(label_string) == 0:
            return None, None
        else:
            idx0 = None
            idx1 = None
            len_query = len(label_string)
            for ind in (i for i, e in enumerate(inp_string) if e == label_string[0]):
                if inp_string[ind : ind + len_query] == label_string:
                    idx0 = ind
                    idx1 = ind + len_query
                    break

            return idx0, idx1
