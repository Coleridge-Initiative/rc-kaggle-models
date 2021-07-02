import logging
import math
import os
import json

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from transformers.utils.dummy_tf_objects import TFRobertaModel

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

from transformers import (
    AutoConfig,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    RobertaTokenizerFast,
    TFAutoModel,
)
from transformers.modeling_tf_utils import load_tf_weights
from transformers.models.bert.modeling_tf_bert import TFBertModel

from callbacks import JaccardFBeta
from data_loader import QueryDataLoader, SupportQueryDataLoader
from model import MetricLearningModel, create_optimizer


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


class BalancedQueryDataLoader(QueryDataLoader):
    """Balanced Query Data sampling for evaluation.

    In this class, the number of negative samples is equal to the number of positive samples.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling()

    def sampling(self):
        num_positive_samples = np.sum(np.array(self.batch_label) != "")
        num_negative_samples = len(self.batch_label) - num_positive_samples
        choosen_positive_idx = np.where(np.array(self.batch_label) != "")[0]
        choosen_negative_idx = np.random.choice(
            list(set(range(len(self.batch_label))) - set(choosen_positive_idx)),
            size=num_positive_samples,
            replace=False,
        )
        choosen_positive_idx = np.random.choice(
            choosen_positive_idx,
            size=int(len(choosen_positive_idx) // self.batch_size) * self.batch_size,
            replace=False,
        )
        choosen_negative_idx = np.random.choice(
            choosen_negative_idx, size=len(choosen_positive_idx), replace=False
        )
        self.num_positive_samples = num_positive_samples
        self.num_negative_samples = num_negative_samples
        self.batch_ids = list(np.array(self.batch_ids)[choosen_positive_idx]) + list(
            np.array(self.batch_ids)[choosen_negative_idx]
        )
        self.batch_text = list(np.array(self.batch_text)[choosen_positive_idx]) + list(
            np.array(self.batch_text)[choosen_negative_idx]
        )
        self.batch_label = list(
            np.array(self.batch_label)[choosen_positive_idx]
        ) + list(np.array(self.batch_label)[choosen_negative_idx])
        self.batch_ids, self.batch_text, self.batch_label = shuffle(
            self.batch_ids, self.batch_text, self.batch_label
        )

    def __len__(self):
        return math.ceil(len(self.batch_text) / self.batch_size)


def make_data_group(data_group, choosen_group_idx, ignore_groups=[0]):
    """Make a new data group from data group with choosen_group_idx and ignore_groups."""

    new_data_group = {}
    for group_idx in choosen_group_idx:
        if group_idx not in ignore_groups:
            new_data_group[group_idx] = data_group[group_idx]
        else:
            print(f"Ignore group {group_idx} for evaluation.")
    return new_data_group


@cli.command("training", short_help="Training")
@click.option(
    "--model_name",
    default="biomed_roberta_base",
    show_default=True,
)
@click.option(
    "--exp_name",
    default="run1",
    show_default=True,
)
def main(model_name, exp_name):
    MODEL_NAME = model_name
    EPOCHS = 10
    BATCH_SIZE = 4
    uncased = False

    settings = json.load(open("./settings.json", "rb"))

    os.makedirs(
        os.path.join(f"{settings['SAVED_MODEL_DIR']}", model_name), exist_ok=True
    )

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
            print("Use scibert uncased version.")
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
    val_df = pd.read_csv(f"./{settings['PROCESSED_DATA_DIR']}/val_sampled.csv")
    train_df = pd.read_csv(f"./{settings['PROCESSED_DATA_DIR']}/train_sampled.csv")

    print("Len train_df: ", len(train_df))
    train_df["text_id"] = np.array(range(len(train_df)))
    val_df["text_id"] = np.array(range(len(val_df)))
    val_query_dataloader = BalancedQueryDataLoader(val_df, batch_size=BATCH_SIZE)
    print("Len QueryDataLoader: ", len(val_query_dataloader))

    norm_fn = None
    if uncased:
        norm_fn = lambda x: x.lower()
    train_dataloader = SupportQueryDataLoader(
        train_df,
        tokenizer=tokenizer,
        batch_size=val_query_dataloader.batch_size,
        is_train=True,
        K=3,  # 3 or 5 is enough
        training_steps=1000,
        support_masked=True,
        query_masked=False,
        query_positive_random_prob=0.5,
        return_query_ids=True,
        return_query_labels=False,
        no_overlap_support_query_group=False,
        allow_support_group_duplicate=True,
        norm_fn=norm_fn,  # keep raw inputs and labels
    )
    valid_dataloader = SupportQueryDataLoader(
        val_df,
        tokenizer=tokenizer,
        batch_size=val_query_dataloader.batch_size,
        is_train=False,
        K=1,  # for faster inference
        return_query_ids=False,
        return_query_labels=False,
        support_data_group=make_data_group(
            train_dataloader.data_group,
            train_dataloader.all_unique_group,
            ignore_groups=[0],
        ),
        training_steps=len(val_query_dataloader),
        query_dataloader=val_query_dataloader,
        support_masked=True,
        query_masked=False,
        no_overlap_support_query_group=train_dataloader.no_overlap_support_query_group,
        allow_support_group_duplicate=train_dataloader.allow_support_group_duplicate,
        norm_fn=train_dataloader.norm_fn,
    )

    optimizer = create_optimizer(
        init_lr=3e-5,
        num_train_steps=train_dataloader.len * (EPOCHS),
        num_warmup_steps=train_dataloader.len * 0.06,
        optimizer_type="adamw",
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)

    config = AutoConfig.from_pretrained(f"./{settings['PRETRAINED_DIR']}/{MODEL_NAME}")
    config.output_attentions = True
    config.output_hidden_states = True

    main_model = TFAutoModel.from_pretrained(
        f"./{settings['PRETRAINED_DIR']}/{MODEL_NAME}", config=config
    )
    model = MetricLearningModel(config=config, name="metric_learning_model")
    model.main_model = main_model
    model.K = train_dataloader.K

    # 1 step call to build model
    support_batch, query_batch = train_dataloader.__getitem__(0)
    support_embeddings, support_mask_embeddings, support_nomask_embeddings = model(
        [
            support_batch["input_ids"],
            support_batch["attention_mask"],
            # support_batch["token_type_ids"],
        ],
        training=True,
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
        training=True,
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

    # compile and fit
    model.compile(
        optimizer=optimizer,
        metric_fn={
            "token_categorical_accuracy": tf.keras.metrics.SparseCategoricalAccuracy(
                name="token_categorical_accuracy"
            ),
            "sentence_categorical_accuracy": tf.keras.metrics.SparseCategoricalAccuracy(
                name="sentence_categorical_accuracy"
            ),
        },
        loss_fn={
            "categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            ),
            "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(
                from_logits=False, reduction=tf.keras.losses.Reduction.NONE
            ),
        },
    )

    model.fit(
        train_dataloader,
        epochs=EPOCHS,
        initial_epoch=0,
        callbacks=[
            JaccardFBeta(
                valid_dataloader,
                f"./{settings['SAVED_MODEL_DIR']}/{MODEL_NAME}/{exp_name}",
                batch_size=64,
                start_epoch=8,
            )
        ],
    )


if __name__ == "__main__":
    cli()
