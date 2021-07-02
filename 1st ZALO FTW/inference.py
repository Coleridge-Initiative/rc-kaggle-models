import logging
import os
import re
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

import click
from transformers import (
    AutoConfig,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    RobertaTokenizerFast,
    TFAutoModel,
)

from data_loader import QueryDataLoader, SupportQueryDataLoader
from model import MetricLearningModel


class PositiveSupportDataLoader(QueryDataLoader):
    """Positive only Support Data Loader."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_idx = np.where(np.array(self.batch_label) != "")[0]
        self.batch_ids = np.array(self.batch_ids)[self.positive_idx]
        self.batch_label = np.array(self.batch_label)[self.positive_idx]
        batch_text = []
        for idx in self.positive_idx:
            batch_text.append(self.batch_text[idx])
        self.batch_text = np.array(batch_text)
        print(f"There are {len(self.positive_idx)} positive samples in training set.")


def clean_text(txt, is_lower=True):
    if is_lower:
        return re.sub("[^A-Za-z0-9]+", " ", str(txt).lower())
    else:
        return re.sub("[^A-Za-z0-9]+", " ", str(txt))


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


@cli.command("extract-support-embedding", short_help="Extract support embeddings.")
@click.option(
    "--model_name",
    default="distilbert-base-uncased",
    show_default=True,
)
@click.option(
    "--pretrained_path",
    default="./saved_model/distilbert-base-uncased/best.h5",
    show_default=True,
)
@click.option(
    "--saved_path",
    default="./saved_model/distilbert-base-uncased/embeddings/",
    show_default=True,
)
@click.option("--batch_size", default=64, show_default=True)
def extract_support_embedding(model_name, pretrained_path, saved_path, batch_size):
    MODEL_NAME = model_name
    uncased = False

    settings = json.load(open("./settings.json", "rb"))

    if "roberta" in MODEL_NAME.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(
            f"./{settings['PRETRAINED_DIR']}/{MODEL_NAME}"
        )
    elif "distilbert" in MODEL_NAME.lower():
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            f"./{settings['PRETRAINED_DIR']}/{MODEL_NAME}"
        )
    elif "scibert" in MODEL_NAME.lower():
        if "uncased" in MODEL_NAME.lower():
            print("Use scibert uncased version")
            uncased = True
        tokenizer = BertTokenizerFast.from_pretrained(
            f"./{settings['PRETRAINED_DIR']}/{MODEL_NAME}", do_lower_case=uncased
        )
    elif "bert" in MODEL_NAME.lower():
        tokenizer = BertTokenizerFast.from_pretrained(
            f"./{settings['PRETRAINED_DIR']}/{MODEL_NAME}"
        )
    else:
        pass
    print(tokenizer)
    train_df = pd.read_csv(f"./{settings['PROCESSED_DATA_DIR']}/train_sampled.csv")
    print("Len: ", len(train_df.iloc[0].text.split()))
    train_df["text_id"] = np.array(range(len(train_df)))
    train_support_dataloader = PositiveSupportDataLoader(
        train_df, batch_size=batch_size
    )
    norm_fn = None
    if uncased:
        norm_fn = lambda x: x.lower()
    valid_dataloader = SupportQueryDataLoader(
        train_df,
        tokenizer=tokenizer,
        batch_size=train_support_dataloader.batch_size,
        is_train=False,
        K=1,
        support_data_group=None,
        training_steps=len(train_support_dataloader),
        query_dataloader=train_support_dataloader,
        support_masked=True,
        query_masked=True,
        no_overlap_support_query_group=True,
        return_query_ids=False,
        norm_fn=norm_fn,
    )

    # Model
    config = AutoConfig.from_pretrained(f"./{settings['PRETRAINED_DIR']}/{MODEL_NAME}")
    config.output_attentions = True
    config.output_hidden_states = True
    main_model = TFAutoModel.from_pretrained(
        f"./{settings['PRETRAINED_DIR']}/{MODEL_NAME}", config=config
    )
    model = MetricLearningModel(config=config, name="metric_learning_model")
    model.main_model = main_model
    model.K = 1

    # 1 step call to build model
    support_batch, query_batch = valid_dataloader.__getitem__(0)
    support_embeddings, support_mask_embeddings, support_nomask_embeddings = model(
        [
            support_batch["input_ids"],
            support_batch["attention_mask"],
            # support_batch["token_type_ids"],
        ],
        training=False,
        sequence_labels=support_batch["sequence_labels"],
    )  # [B, F]
    (
        query_embeddings,
        query_mask_embeddings,
        query_nomask_embeddings,
        attention_values,
    ) = model(
        [
            query_batch["input_ids"],
            query_batch["attention_mask"],
            # query_batch["token_type_ids"],
        ],
        training=False,
        sequence_labels=None,
        mask_embeddings=support_mask_embeddings,
        nomask_embeddings=support_nomask_embeddings,
    )  # [B, F]
    positive_idx = tf.where(tf.math.equal(query_batch["classes"], 1))[..., 0]
    mask_nomask_embeddings = tf.concat(
        [
            support_mask_embeddings,
            tf.gather(query_mask_embeddings, positive_idx),
            support_nomask_embeddings,
        ],
        axis=0,
    )
    classes = tf.concat(
        [
            tf.ones(shape=[tf.shape(support_mask_embeddings)[0]], dtype=tf.int32),
            tf.ones(shape=[tf.shape(positive_idx)[0]], dtype=tf.int32),
            tf.zeros(shape=[tf.shape(support_nomask_embeddings)[0]], dtype=tf.int32),
        ],
        axis=0,
    )
    token_logits = model.token_arcface([mask_nomask_embeddings, classes])
    sentence_embeddings = tf.concat([support_embeddings, query_embeddings], axis=0)
    sentence_classes = tf.concat(
        [support_batch["classes"], query_batch["classes"]], axis=0
    )
    sentence_logits = model.sentence_arcface([sentence_embeddings, sentence_classes])
    print(mask_nomask_embeddings.shape)
    print(classes.shape)

    # Load weights
    model.load_weights(pretrained_path, by_name=True)
    model.summary()

    model = tf.function(model, experimental_relax_shapes=True)

    # Extract support embedding from training set
    # Note that query in dataload is support in this case
    os.makedirs(saved_path, exist_ok=True)
    all_support_embeddings = []
    all_support_mask_embeddings = []
    all_support_nomask_embeddings = []
    for _, support_batch in tqdm(valid_dataloader):
        support_embeddings, support_mask_embeddings, support_nomask_embeddings = model(
            [
                support_batch["input_ids"],
                support_batch["attention_mask"],
                # support_batch["token_type_ids"],
            ],
            training=False,
            sequence_labels=support_batch["sequence_labels"],
        )  # [B, F]
        all_support_embeddings.append(support_embeddings.numpy())
        all_support_mask_embeddings.append(support_mask_embeddings.numpy())
        all_support_nomask_embeddings.append(support_nomask_embeddings.numpy())

    # Concatenate and save to file
    all_support_embeddings = np.concatenate(all_support_embeddings, axis=0)
    all_support_mask_embeddings = np.concatenate(all_support_mask_embeddings, axis=0)
    all_support_nomask_embeddings = np.concatenate(
        all_support_nomask_embeddings, axis=0
    )

    print("Support Embedding Shape: ", all_support_embeddings.shape)
    print("Support Mask Embedding Shape: ", all_support_mask_embeddings.shape)
    print("Support NoMask Embedding Shape: ", all_support_nomask_embeddings.shape)

    # Save to file
    np.save(os.path.join(saved_path, "support_embeddings.npy"), all_support_embeddings)
    np.save(
        os.path.join(saved_path, "support_mask_embeddings.npy"),
        all_support_mask_embeddings,
    )
    np.save(
        os.path.join(saved_path, "support_nomask_embeddings.npy"),
        all_support_nomask_embeddings,
    )


if __name__ == "__main__":
    cli()
