import os
import re

import nltk
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tqdm import tqdm

nltk.download("popular")

import json
import os
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

os.environ["TOKENIZERS_PARALLELISM"] = "false"

stop_words = set(stopwords.words("english"))


def remove_stopwords(string):
    word_tokens = word_tokenize(string)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence).strip()


def remove_first_stopword(string):
    word_tokens = string.split(" ")
    stop_idx = 0
    for i in range(len(word_tokens)):
        if word_tokens[i] in stop_words:
            stop_idx += 1
        else:
            break
    return " ".join(word_tokens[stop_idx:])


def jaccard_similarity(str1, str2):
    a = set(str1.lower().split(" "))
    b = set(str2.lower().split(" "))
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_cosine_similarity(x1, x2):
    x1_norm = tf.nn.l2_normalize(x1, axis=1)
    x2_norm = tf.nn.l2_normalize(x2, axis=1)
    cosine_similarity = tf.matmul(x1_norm, x2_norm, transpose_b=True)  # [B1, B2]
    return tf.clip_by_value(cosine_similarity, -1.0, 1.0)


def find_all_start_end(attention_values):
    start_offset = {}
    current_idx = 0
    is_start = False
    start_end = []
    while current_idx < len(attention_values):
        if attention_values[current_idx] == 1 and is_start is False:
            start_offset[current_idx] = 0
            is_start = True
            start_idx = current_idx
        elif attention_values[current_idx] == 1 and is_start is True:
            start_offset[start_idx] += 1
        elif attention_values[current_idx] == 0 and is_start is True:
            is_start = False
        current_idx += 1
    for k, v in start_offset.items():
        start_end.append([k, k + v + 1])
    return start_end


def check_number(l):
    ws = l.split(" ")
    for w in ws:
        if w.isnumeric():
            return False
    return True


def clean_text_v2(txt):
    return re.sub("[^A-Za-z0-9\(\)]+", " ", str(txt).lower())


def check_valid_acronym(label, acronym):
    guess_acronym = " ".join([w[0] for w in label.split()])
    js = jaccard_similarity(guess_acronym, " ".join([c for c in acronym.split()[0]]))
    if js >= 0.5:
        return True
    return False


def is_last_word_acronym(label):
    words = label.split()
    last_word = words[-1]
    label = " ".join(words[:-1])
    if check_valid_acronym(label, last_word):
        return True
    return False


def find_cased_pred(
    lower_start_idx, lower_end_idx, lower_string, cased_string, lower_pred
):
    len_lower_string = len(lower_string)
    len_cased_string = len(cased_string)
    if (len_lower_string - len_cased_string) == 0:
        return cased_string[lower_start_idx:lower_end_idx]
    else:
        diff_len = abs(len_lower_string - lower_end_idx)
        for shift_idx in range(-diff_len - 1, diff_len + 1):
            cased_pred_candidate = cased_string[
                lower_start_idx
                + shift_idx : lower_start_idx
                + shift_idx
                + len(lower_pred)
            ]
            if cased_pred_candidate.lower() == lower_pred:
                return cased_pred_candidate
    return lower_pred.capitalize()


def calculate_iou(se_0, se_1):
    s_0, e_0 = se_0
    s_1, e_1 = se_1
    max_s = max(s_0, s_1)
    min_e = min(e_0, e_1)
    intersection = min_e - max_s
    return intersection / ((e_0 - s_0) + (e_1 - s_1) - intersection)


def find_all_pred_in_text(text, all_unique_preds, return_raw_pred=False):
    # text = unicode_tr(text)
    text_cased = clean_text(text, False).strip()
    text = clean_text(text).strip()
    preds = []
    preds_indexs = []
    for pred in all_unique_preds:
        if pred in text and pred != "":
            preds.append(pred)
    unique_preds = []  # unique in terms of index.
    unique_raw_preds = []
    for pred in preds:
        matchs = re.finditer(pred, text)
        for match in matchs:
            start_index = match.start()
            end_index = match.end()
            pred_cased = find_cased_pred(start_index, end_index, text, text_cased, pred)
            if pred_cased.islower() is False:
                preds_indexs.append([start_index, end_index])
                unique_preds.append(pred)
                unique_raw_preds.append(pred_cased)
                break  # only get the first for the evaluation.
    group_idxs = []
    for i in range(len(preds_indexs)):
        for j in range(len(preds_indexs)):
            if i != j:
                start_i, end_i = preds_indexs[i]
                start_j, end_j = preds_indexs[j]
                if start_i <= end_j and end_i <= end_j and start_i >= start_j:
                    group_idxs.append([i, j])
    unique_preds = np.array(unique_preds)
    unique_raw_preds = np.array(unique_raw_preds)
    for group_idx in group_idxs:
        unique_preds[group_idx[0]] = unique_preds[group_idx[1]]
        unique_raw_preds[group_idx[0]] = unique_raw_preds[group_idx[1]]
    if return_raw_pred:
        return np.unique(unique_preds), np.unique(unique_raw_preds)
    else:
        return np.unique(unique_preds)


def clean_text(txt, is_lower=True):
    if is_lower:
        return re.sub("[^A-Za-z0-9]+", " ", str(txt).lower())
    else:
        return re.sub("[^A-Za-z0-9]+", " ", str(txt))


def compute_fbeta(y_true, y_pred, beta=0.5, cosine_th=0.2):
    """Compute the Jaccard-based micro FBeta score.

    References
    ----------
    - https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/overview/evaluation
    """
    import copy

    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative
    fp_list = []
    y_true_copy = copy.deepcopy(y_true)
    y_pred_copy = copy.deepcopy(y_pred)
    for ground_truth_list, predicted_string_list in zip(y_true_copy, y_pred_copy):
        predicted_string_list_sorted = sorted(predicted_string_list)
        n_predicted_string = len(predicted_string_list_sorted)
        n_gt_string = len(ground_truth_list)
        if n_gt_string > n_predicted_string:
            fn += n_gt_string - n_predicted_string
        elif n_gt_string < n_predicted_string:
            fp += n_predicted_string - n_gt_string
            fp_list.extend(
                predicted_string_list_sorted[-(n_predicted_string - n_gt_string) :]
            )

        start_idx = 0
        N = min(n_gt_string, n_predicted_string)
        while start_idx < N:
            # find nearest groundtruth to match with predicted_string
            predicted_string = predicted_string_list_sorted[start_idx]
            jaccard_with_gt = [
                jaccard_similarity(predicted_string, ground_truth_list[i])
                for i in range(len(ground_truth_list))
            ]
            best_matched_gt_idx = np.argmax(jaccard_with_gt)
            if jaccard_with_gt[best_matched_gt_idx] >= 0.5:
                tp += 1
            else:
                fp += 1
                fp_list.append(predicted_string)
            start_idx += 1
            ground_truth_list.pop(best_matched_gt_idx)

    raw_values = [tp, fp, fn]

    tp *= 1 + beta ** 2
    fn *= beta ** 2
    fbeta_score = tp / (tp + fp + fn)
    return fbeta_score, raw_values, fp_list


def find_longest_group(preds, jaccard_th=0.5, jaccard_matrix=None, choosen_idxs=None):
    mask = np.zeros(shape=[len(preds)], dtype=np.float32)
    preds = preds[choosen_idxs]
    for i in choosen_idxs:
        mask[i] = 1.0
    longest_candidates_idx = []
    candidates_score = [
        np.mean(jaccard_matrix[choosen_idxs[i], ...] * mask) for i in range(len(preds))
    ]
    best_candidates_idx = np.argmax(candidates_score)
    longest_candidates_idx.append(best_candidates_idx)
    while True:
        old_best_candidates_idx = best_candidates_idx
        jaccard_ = []
        for i in range(len(preds)):
            if i not in longest_candidates_idx:
                jaccard_i = []
                for k in longest_candidates_idx:
                    jaccard_i.append(jaccard_matrix[choosen_idxs[i], choosen_idxs[k]])
                jaccard_i = np.min(jaccard_i)
                jaccard_.append(jaccard_i)
            else:
                jaccard_.append(-1.0)
        best_candidates_idx = np.argmax(jaccard_)
        if jaccard_[best_candidates_idx] >= jaccard_th:
            longest_candidates_idx.append(best_candidates_idx)
        else:
            break
    return longest_candidates_idx


def find_best_sample_in_group(group, jaccard_similarity_dict=None):
    jaccard_matrix = np.zeros(shape=[len(group), len(group)], dtype=np.float32)
    for i in range(jaccard_matrix.shape[0]):
        for j in range(jaccard_matrix.shape[1]):
            if i != j:
                jaccard_matrix[i, j] = jaccard_similarity_dict[
                    group[i] + "|" + group[j]
                ]
            else:
                jaccard_matrix[i, j] = 0.0
    avg_jaccard = np.mean(jaccard_matrix, axis=1)
    return group[np.argmax(avg_jaccard)]


def find_all_group(preds, jaccard_th=0.5, maximum_group_steps=50):
    all_idxs = np.array(list(range(len(preds))))
    all_group = {}
    remain_idxs = all_idxs
    #     print("Len preds: ", len(preds))
    jaccard_similarity_dict = {}
    # pre-compute jaccard matrix dict
    jaccard_matrix = -1.0 * np.ones(shape=[len(preds), len(preds)], dtype=np.float32)
    for i in range(jaccard_matrix.shape[0]):
        for j in range(jaccard_matrix.shape[1]):
            if i != j:
                if jaccard_matrix[j, i] == -1.0:
                    jaccard_matrix[i, j] = jaccard_similarity(preds[i], preds[j])
                    jaccard_matrix[j, i] = jaccard_matrix[i, j]
                else:
                    jaccard_matrix[i, j] = jaccard_matrix[j, i]
            else:
                jaccard_matrix[i, j] = 0.0
            if preds[i] + "|" + preds[j] not in jaccard_similarity_dict:
                jaccard_similarity_dict[preds[i] + "|" + preds[j]] = jaccard_matrix[
                    i, j
                ]
    #     print("Finishing calculate jaccard similarity matrix")
    steps = 0
    while True:
        if steps <= maximum_group_steps:
            longest_candidates_idx = find_longest_group(
                preds,
                jaccard_th=jaccard_th,
                jaccard_matrix=jaccard_matrix,
                choosen_idxs=remain_idxs,
            )
            group = preds[remain_idxs][longest_candidates_idx]
            if len(longest_candidates_idx) >= 3:
                best_sample_in_group = find_best_sample_in_group(
                    group=group, jaccard_similarity_dict=jaccard_similarity_dict
                )
            elif len(longest_candidates_idx) == 2:
                best_sample_in_group = group[np.argmax([len(g) for g in group])]
            else:
                best_sample_in_group = group[np.argmax([len(g) for g in group])]
            steps += 1
        else:
            longest_candidates_idx = [0]
            group = preds[remain_idxs][longest_candidates_idx]
            best_sample_in_group = group[np.argmax([len(g) for g in group])]

        if best_sample_in_group not in all_group:
            all_group[best_sample_in_group] = group
        remain_idxs = np.array(
            list(set(remain_idxs) - set(remain_idxs[longest_candidates_idx]))
        )
        #         print(len(remain_idxs))
        if len(remain_idxs) == 0:
            break
    return all_group


def find_unique_preds(preds, jaccard_threshold=0.5, maximum_group_steps=50):
    all_preds = np.array(preds)
    all_sigle_preds = []
    for pred in all_preds:
        all_sigle_preds.extend(pred.split("|"))

    def check_number(l):
        ws = l.split(" ")
        for w in ws:
            if w.isnumeric():
                return False
        return True

    all_sigle_preds = np.array(np.unique(all_sigle_preds))
    all_sigle_preds = np.array(list(set(all_sigle_preds) - set([""])))
    all_unique_preds = find_all_group(
        all_sigle_preds, jaccard_threshold, maximum_group_steps
    )
    all_groups = all_unique_preds
    all_unique_preds = list(all_unique_preds.keys())
    return all_unique_preds, all_groups


class JaccardFBeta(tf.keras.callbacks.Callback):
    """A callback to calculate JaccardFbeta for the valid set

    Args:
        valid_dataloader: A valid dataloader
        saved_path: A saved directory for trained models
        batch_size: A batch size for valid dataloader
        start_epoch: An epoch index which start to run this callback
    """
    def __init__(
        self, valid_dataloader, saved_path, batch_size=64, start_epoch=8, **kwargs
    ):
        super().__init__(**kwargs)

        logs = {}
        self.start_epoch = start_epoch
        with open(os.path.join(saved_path, "logs.json"), "w") as out_file:
            json.dump(logs, out_file)

        self.valid_dataloader = valid_dataloader
        self.saved_path = saved_path
        self.tokenizer = self.valid_dataloader.tokenizer
        self.val_df = self.valid_dataloader.data

        # reset some params
        self.valid_dataloader.batch_size = batch_size
        self.valid_dataloader.query_dataloader.batch_size = batch_size
        self.valid_dataloader.donot_process_support = False
        self.valid_dataloader.return_query_ids = True
        self.valid_dataloader.return_query_labels = True

        # create text per id
        self.raw_text_per_id = {}
        self.gt_group_label_per_id = {}
        self.all_unique_ids = self.val_df.id.unique()

        if os.path.isfile(os.path.join("./saved_models", "raw_text_per_id.json")):
            self.raw_text_per_id = json.load(
                open(os.path.join("./saved_models", "raw_text_per_id.json"), "rb")
            )
        else:
            for id in tqdm(self.all_unique_ids, desc="Create raw text per id"):
                full_text = " ".join(
                    self.val_df[self.val_df.id == id].text.tolist()
                ).strip()
                if id not in self.raw_text_per_id:
                    self.raw_text_per_id[id] = full_text

        try:
            self.all_valid_labels = np.load(
                os.path.join("./saved_models", "all_valid_labels.npy")
            )
            self.gt_group_label_per_id = json.load(
                open(os.path.join("./saved_models", "gts.json"), "rb")
            )
        except:
            all_valid_labels = []
            for i in tqdm(
                range(len(self.all_unique_ids)), desc="Create GT label per id"
            ):
                full_text = self.raw_text_per_id[self.all_unique_ids[i]]
                all_labels = self.val_df[
                    self.val_df.id == self.all_unique_ids[i]
                ].label.tolist()
                all_single_labels = []
                for l in all_labels:
                    if isinstance(l, str):
                        preds = l.split("|")
                        for pred in preds:
                            if pred != "":
                                all_single_labels.append(clean_text(pred).strip())
                                all_valid_labels.extend(all_single_labels)

                merged_gt_labels = find_all_pred_in_text(
                    full_text, np.unique(all_single_labels)
                )
                self.gt_group_label_per_id[self.all_unique_ids[i]] = []
                self.gt_group_label_per_id[self.all_unique_ids[i]].extend(
                    merged_gt_labels
                )

            with open(os.path.join("./saved_models", "gts.json"), "w") as out_file:
                json.dump(self.gt_group_label_per_id, out_file)

            with open(
                os.path.join("./saved_models", "raw_text_per_id.json"), "w"
            ) as out_file:
                json.dump(self.raw_text_per_id, out_file)

            self.all_valid_labels = np.unique(all_valid_labels)
            np.save(
                os.path.join("./saved_models", "all_valid_labels.npy"),
                self.all_valid_labels,
            )
            print(f"There are {len(self.all_valid_labels)} unique validation labels.")

    def get_fp_preds(self, fp_list):
        fp_preds = []
        fp_list_counter = Counter(fp_list)
        labels = []
        counts = []
        for l, c in fp_list_counter.items():
            labels.append(l)
            counts.append(c)

        sort_idxs = np.argsort(counts)[::-1]
        counts = list(np.array(counts)[sort_idxs])
        labels = list(np.array(labels)[sort_idxs])

        for l, c in zip(labels, counts):
            fp_preds.append(f"{l}: {c}")
        return fp_preds

    def on_epoch_end(self, epoch, logs):
        if epoch >= self.start_epoch:
            self._on_epoch_end(epoch, logs)

    def _on_epoch_end(self, epoch, logs):
        # set model.K to valid_dataloader K
        train_K = self.model.K
        self.model.K = self.valid_dataloader.K
        inference_model = tf.function(self.model, experimental_relax_shapes=True)
        preds_dict = {}
        TH_LIST = [0.5, 0.55, 0.6, 0.65, 0.7]
        for th in TH_LIST:
            preds_dict[th] = []
        gts = []
        cosines = []
        ids = []
        inputs = []
        # len(self.valid_dataloader.query_dataloader)
        for i in tqdm(range(len(self.valid_dataloader.query_dataloader) - 1)):
            all_preds = []
            all_gts = []
            support_batch, query_batch = self.valid_dataloader.__getitem__(i)
            (
                support_embeddings,
                support_mask_embeddings,
                support_nomask_embeddings,
            ) = inference_model(
                [support_batch["input_ids"], support_batch["attention_mask"]],
                training=False,
                sequence_labels=support_batch["sequence_labels"],
            )  # [B, F]
            (
                query_embeddings,
                query_mask_embeddings,
                query_nomask_embeddings,
                attention_values,
            ) = inference_model(
                [
                    query_batch["input_ids"],
                    query_batch["attention_mask"],
                ],
                training=False,
                mask_embeddings=support_mask_embeddings,
                nomask_embeddings=support_nomask_embeddings,
            )  # [B, F]
            cosine = compute_cosine_similarity(
                query_embeddings, support_embeddings
            ).numpy()
            cosine = np.mean(cosine, axis=1)
            all_preds.extend(cosine)
            all_gts.extend(query_batch["classes"])
            ids.extend(query_batch["ids"])
            gts.extend(query_batch["labels"])
            for k in range(len(all_gts)):
                binary_pred = attention_values.numpy()[k, :, 0]
                inputs.append(self.tokenizer.decode(query_batch["input_ids"][k, ...]))
                for TH in TH_LIST:
                    binary_pred_at_th = binary_pred >= TH
                    if np.sum(binary_pred_at_th) > 0:
                        binary_pred_at_th = binary_pred_at_th.astype(np.int32)
                        start_end = find_all_start_end(binary_pred_at_th)
                        pred_candidates = []
                        for s_e in start_end:
                            if (s_e[1] - s_e[0]) >= 4:
                                pred_tokens = list(range(s_e[0], s_e[1]))
                                pred = self.tokenizer.decode(
                                    query_batch["input_ids"][k, ...][pred_tokens]
                                )
                                pred_candidates.append(pred)
                        pred = "|".join(pred_candidates)
                    else:
                        pred = ""

                    preds_dict[TH].append(pred)
                cosines.append(all_preds[k])

        # Calculate best F1 score
        all_gts = (np.array(gts) != "").astype(np.int32)
        all_preds = np.copy(cosines)
        all_gts = all_gts[: len(cosines)]
        best_f1 = -100
        best_th = None
        for th in np.linspace(-0.9, 0.9, 100):
            all_preds_at_th = list((np.array(all_preds) >= th).astype(np.int32))
            f1 = f1_score(all_gts, all_preds_at_th, average="macro")
            if f1 >= best_f1:
                best_f1 = f1
                best_th = th

        print(f"Best F1 Score ({best_f1:.3f}) at Threshold ({best_th:.3f})")

        jaccard_fbetas = []

        current_logs = json.load(open(os.path.join(self.saved_path, "logs.json"), "rb"))
        if epoch not in current_logs:
            current_logs[epoch] = {}

        for TH in TH_LIST:
            preds = preds_dict[TH]
            # Get all accepted preds
            all_accepted_preds = []

            for i in range(len(preds)):
                if cosines[i] >= best_th:
                    a = preds[i].split("|")
                    unique_v = np.unique(a)
                    all_accepted_preds.extend(unique_v)

            accepted_preds = []
            for k in all_accepted_preds:
                k = remove_first_stopword(clean_text(k).strip())
                if (
                    "#" not in k
                    and "ġ" not in k
                    and len(k.strip().split(" ")) >= 3
                    and len(clean_text(k).strip().split(" ")) >= 3
                    and len(remove_stopwords(k).split(" ")) >= 3
                    and len(k) >= 10
                    and len(k) <= 100
                    and check_number(k)
                ):
                    accepted_preds.append(clean_text(k).strip())

            counter_accepted_preds = Counter(accepted_preds)
            accepted_preds = list(set(accepted_preds) - set([""]))
            accepted_preds = accepted_preds
            print("Len All preds: ", len(list(set(all_accepted_preds))))
            print("Len Accepted preds: ", len(accepted_preds))

            if len(accepted_preds) <= 10:
                need_break = True
            else:
                need_break = False

            gt_group_label_per_id = self.gt_group_label_per_id
            group_label_per_id = {}

            for i in range(len(self.all_unique_ids)):
                full_text = self.raw_text_per_id[self.all_unique_ids[i]]
                merged_pred_labels = find_all_pred_in_text(
                    full_text, accepted_preds, False
                )
                if len(merged_pred_labels) >= 2:
                    merged_pred_labels = find_unique_preds(
                        merged_pred_labels, 0.5, 500
                    )[0]
                group_label_per_id[self.all_unique_ids[i]] = []
                group_label_per_id[self.all_unique_ids[i]].extend(merged_pred_labels)

            for k, v in gt_group_label_per_id.items():
                unique_v = list(np.unique(v))
                if len(unique_v) >= 2:
                    gt_group_label_per_id[k] = "|".join(
                        [v for v in unique_v if v != ""]
                    )
                elif len(unique_v) == 1 and unique_v[0] == "":
                    gt_group_label_per_id[k] = ""
                elif len(unique_v) == 1 and unique_v[0] != "":
                    gt_group_label_per_id[k] = f"{unique_v[0]}"
                else:
                    gt_group_label_per_id[k] = ""

            for k, v in group_label_per_id.items():
                unique_v = list(np.unique(v))
                if len(unique_v) >= 2:
                    group_label_per_id[k] = "|".join([v for v in unique_v if v != ""])
                elif len(unique_v) == 1 and unique_v[0] == "":
                    group_label_per_id[k] = ""
                elif len(unique_v) == 1 and unique_v[0] != "":
                    group_label_per_id[k] = f"{unique_v[0]}"
                else:
                    group_label_per_id[k] = ""

            y_true = []
            y_pred = []

            for k in list(gt_group_label_per_id.keys()):
                y_true_ = gt_group_label_per_id[k].split("|")
                y_true_ = [clean_text(y).strip() for y in y_true_]
                y_true_ = list(set(y_true_) - set([""]))
                y_true.append(y_true_)
                pred = []
                pred.extend(group_label_per_id[k].split("|"))
                pred = list(set(pred) - set([""]))
                accepted_pred = []
                for i in range(len(pred)):
                    clean_pred = clean_text(pred[i])
                    pred[i] = clean_pred.strip()
                    accepted_pred.append(pred[i])
                y_pred.append(list(pred))

            jaccard_fbeta, [tp, fp, fn], fp_list = compute_fbeta(y_true, y_pred)
            jaccard_fbetas.append(jaccard_fbeta)

            print(
                f"Jaccard-based FBeta (0.5) @ Threshold ({TH:.2f}): {jaccard_fbeta:.3f} (n_preds: {len(accepted_preds)})"
            )
            # print("==============================================")

            if need_break:
                break

            # write into log
            if TH not in current_logs[epoch]:
                current_logs[epoch][TH] = {}
            current_logs[epoch][TH]["TP"] = tp
            current_logs[epoch][TH]["FP"] = fp
            current_logs[epoch][TH]["FN"] = fn
            current_logs[epoch][TH]["JaccardF0.5"] = jaccard_fbeta
            current_logs[epoch][TH]["n_preds"] = len(accepted_preds)
            current_logs[epoch][TH]["fp_list"] = self.get_fp_preds(fp_list)

        with open(os.path.join(self.saved_path, "logs.json"), "w") as out_file:
            json.dump(current_logs, out_file)

        if len(jaccard_fbetas) >= 1:
            best_jaccard_fbeta = np.max(jaccard_fbetas)
            best_jaccard_fbeta_threshold = TH_LIST[np.argmax(jaccard_fbetas)]
            avg_jaccard_fbeta = np.mean(jaccard_fbetas)
            print(f"Average Jaccard-based FBeta (0.5): {avg_jaccard_fbeta:.3f}")
            self.model.save_weights(
                os.path.join(
                    self.saved_path,
                    f"{epoch}-{best_jaccard_fbeta:.3f}-{avg_jaccard_fbeta:.3f}-{best_jaccard_fbeta_threshold:.2f}-{best_th:.3f}.h5",
                )
            )

        # reset K
        self.model.K = train_K
