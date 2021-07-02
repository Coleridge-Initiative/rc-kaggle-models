import tensorflow as tf
import tensorflow_addons as tfa
from transformers import *
from transformers.optimization_tf import AdamWeightDecay, WarmUp

from metric_layers import ArcFace


def create_optimizer(
    init_lr, num_train_steps, num_warmup_steps, optimizer_type="adamw"
):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        end_learning_rate=1e-5,
    )
    if num_warmup_steps:
        learning_rate_fn = WarmUp(
            initial_learning_rate=init_lr,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=num_warmup_steps,
        )

    if optimizer_type == "adamw":
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )
    elif optimizer_type == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_fn, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )
    elif optimizer_type == "lamb":
        optimizer = tfa.optimizers.LAMB(
            learning_rate=learning_rate_fn,
            weight_decay_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )
    return optimizer


class MetricLearningModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.main_model = None
        self.support_dense = tf.keras.layers.Dense(
            units=768, activation=None, dtype=tf.float32
        )
        self.config = config
        self.K = 3
        self.token_arcface = ArcFace(
            2, margin=0.5, logist_scale=10, easy_margin=True, centers=1
        )
        self.sentence_arcface = self.token_arcface

    def _compute_avg_embeddings(self, sequence_embeddings, attentions_mask, K=3):
        embeddings = tf.reduce_mean(
            attentions_mask * sequence_embeddings, axis=1
        )  # [B * K, F]
        if K > 1:
            embeddings = tf.reshape(
                embeddings,
                (-1, K, self.support_dense.units),
            )
            embeddings = tf.reduce_mean(embeddings, axis=1)  # [B, F]
        return embeddings

    def call(
        self,
        inputs,
        training=False,
        sequence_labels=None,
        mask_embeddings=None,
        nomask_embeddings=None,
    ):
        output_hidden_states = self.main_model(
            input_ids=inputs[0], attention_mask=inputs[1], training=training
        )[-2]
        concat_hidden_states = tf.concat(
            output_hidden_states[-1:], axis=-1
        )  # [B * K, T, F]
        concat_hidden_states = self.support_dense(
            concat_hidden_states
        )  # [B * K, T, 768]
        sequence_embeddings = concat_hidden_states[:, 0, :]  # [B * K, 768]
        if sequence_labels is not None:
            sequence_labels = tf.cast(
                sequence_labels, dtype=concat_hidden_states.dtype
            )[..., None]
            mask_embeddings = self._compute_avg_embeddings(
                concat_hidden_states,
                tf.where(sequence_labels == -100, 0.0, sequence_labels),
                self.K,
            )
            nomask_embeddings = self._compute_avg_embeddings(
                concat_hidden_states,
                1.0 - tf.where(sequence_labels == -100, 1.0, sequence_labels),
                K=self.K,
            )
            return sequence_embeddings, mask_embeddings, nomask_embeddings
        else:
            attention_mask = tf.cast(inputs[1], concat_hidden_states.dtype)[
                ..., None
            ]  # [B, T, 1]
            normed_mask_embeddings = tf.nn.l2_normalize(mask_embeddings, axis=1)[
                ..., None
            ]
            normed_nomask_embeddings = tf.nn.l2_normalize(nomask_embeddings, axis=1)[
                ..., None
            ]
            normed_hidden_states = tf.nn.l2_normalize(concat_hidden_states, axis=-1)
            mask_cosine_similarity = tf.matmul(
                normed_hidden_states, normed_mask_embeddings
            )  # [B, T, 1]
            nomask_cosine_similarity = tf.matmul(
                normed_hidden_states, normed_nomask_embeddings
            )  # [B, T, 1]
            mask_attentions = tf.nn.sigmoid(10.0 * mask_cosine_similarity)  # [B, T, 1]
            nomask_attentions = tf.nn.sigmoid(
                10.0 * nomask_cosine_similarity
            )  # [B, T, 1]

            # average attention
            # you can only use 1.0 * mask_attentions, the end result is didn't change much.
            attentions = 0.5 * (mask_attentions + (1.0 - nomask_attentions))

            # mask
            attentions *= attention_mask

            # compute mask and nomask embeddings
            mask_embeddings = self._compute_avg_embeddings(
                concat_hidden_states,
                tf.where(attention_mask == 0, 0.0, attentions),
                K=1,
            )
            nomask_embeddings = self._compute_avg_embeddings(
                concat_hidden_states,
                1.0 - tf.where(attention_mask == 0, 1.0, attentions),
                K=1,
            )
            return sequence_embeddings, mask_embeddings, nomask_embeddings, attentions

    def _apply_gradients(self, total_loss):
        # compute gradient
        if isinstance(
            self.optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer
        ):
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)
        else:
            scaled_loss = total_loss
        scaled_gradients = tf.gradients(scaled_loss, self.trainable_variables)
        if isinstance(
            self.optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer
        ):
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def compile(self, optimizer, metric_fn, loss_fn):
        super().compile(optimizer)
        self.metric_fn = metric_fn
        self.loss_fn = loss_fn

    def compute_sequence_loss(self, labels, preds):
        active_loss = tf.reshape(labels, (-1,)) != -100
        reduced_preds = tf.boolean_mask(tf.reshape(preds, (-1,)), active_loss)
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        labels = tf.cast(labels, reduced_preds.dtype)
        return tf.reduce_mean(
            self.loss_fn["binary_crossentropy"](
                labels[:, None],
                reduced_preds[:, None],
            )
        )

    @tf.function
    def train_step(self, data):
        support_batch, query_batch = data
        support_embeddings, support_mask_embeddings, support_nomask_embeddings = self(
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
        ) = self(
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
        attention_loss = self.compute_sequence_loss(
            query_batch["sequence_labels"],
            tf.squeeze(attention_values, -1),
        )

        # arcface loss
        positive_idx = tf.where(tf.math.equal(query_batch["classes"], 1))[..., 0]
        negative_idx = tf.where(tf.math.equal(query_batch["classes"], 0))[..., 0]
        mask_nomask_embeddings = tf.concat(
            [
                support_mask_embeddings,  # positive_support_mask
                tf.gather(query_mask_embeddings, positive_idx),  # positive_query_mask
                support_nomask_embeddings,  # positive_support_nomask
            ],
            axis=0,
        )
        classes = tf.concat(
            [
                tf.ones(shape=[tf.shape(support_mask_embeddings)[0]], dtype=tf.int32),
                tf.ones(shape=[tf.shape(positive_idx)[0]], dtype=tf.int32),
                tf.zeros(
                    shape=[tf.shape(support_nomask_embeddings)[0]], dtype=tf.int32
                ),
            ],
            axis=0,
        )
        token_logits = self.token_arcface([mask_nomask_embeddings, classes])
        token_arcface_loss = tf.reduce_mean(
            self.loss_fn["categorical_crossentropy"](classes, token_logits)
        )
        self.metric_fn["token_categorical_accuracy"].update_state(
            tf.expand_dims(classes, -1),
            tf.nn.softmax(token_logits, axis=-1),
        )
        # sentence embeddings loss
        sentence_embeddings = tf.concat([support_embeddings, query_embeddings], axis=0)
        sentence_classes = tf.concat(
            [support_batch["classes"], query_batch["classes"]], axis=0
        )
        sentence_logits = self.sentence_arcface([sentence_embeddings, sentence_classes])
        sentence_arcface_loss = tf.reduce_mean(
            self.loss_fn["categorical_crossentropy"](
                sentence_classes,
                sentence_logits,
            )
        )
        self.metric_fn["sentence_categorical_accuracy"].update_state(
            tf.expand_dims(sentence_classes, -1),
            tf.nn.softmax(sentence_logits, axis=-1),
        )

        loss = attention_loss + token_arcface_loss + sentence_arcface_loss
        self._apply_gradients(loss)
        results = {}
        results["loss"] = loss
        results["attention_loss"] = attention_loss
        results["token_arcface_loss"] = token_arcface_loss
        results["sentence_arcface_loss"] = sentence_arcface_loss
        results["token_arcface_accuracy"] = self.metric_fn[
            "token_categorical_accuracy"
        ].result()
        results["sentence_arcface_accuracy"] = self.metric_fn[
            "sentence_categorical_accuracy"
        ].result()
        return results

    @tf.function
    def test_step(self, data):
        support_batch, query_batch = data
        support_embeddings, support_mask_embeddings, support_nomask_embeddings = self(
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
        ) = self(
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
        attention_loss = self.compute_sequence_loss(
            query_batch["sequence_labels"], tf.squeeze(attention_values, -1)
        )
        # arcface loss
        positive_idx = tf.where(tf.math.equal(query_batch["classes"], 1))[..., 0]
        negative_idx = tf.where(tf.math.equal(query_batch["classes"], 0))[..., 0]
        mask_nomask_embeddings = tf.concat(
            [
                support_mask_embeddings,
                tf.gather(query_mask_embeddings, positive_idx),  # positive_query_mask
                support_nomask_embeddings,
            ],
            axis=0,
        )
        classes = tf.concat(
            [
                tf.ones(shape=[tf.shape(support_mask_embeddings)[0]], dtype=tf.int32),
                tf.ones(shape=[tf.shape(positive_idx)[0]], dtype=tf.int32),
                tf.zeros(
                    shape=[tf.shape(support_nomask_embeddings)[0]], dtype=tf.int32
                ),
            ],
            axis=0,
        )
        token_logits = self.token_arcface([mask_nomask_embeddings, classes])
        token_arcface_loss = tf.reduce_mean(
            self.loss_fn["categorical_crossentropy"](classes, token_logits)
        )
        self.metric_fn["token_categorical_accuracy"].update_state(
            tf.expand_dims(classes, -1),
            tf.nn.softmax(token_logits, axis=-1),
        )
        # sentence embeddings loss
        sentence_embeddings = tf.concat([support_embeddings, query_embeddings], axis=0)
        sentence_classes = tf.concat(
            [support_batch["classes"], query_batch["classes"]], axis=0
        )
        sentence_logits = self.sentence_arcface([sentence_embeddings, sentence_classes])
        sentence_arcface_loss = tf.reduce_mean(
            self.loss_fn["categorical_crossentropy"](sentence_classes, sentence_logits)
        )
        self.metric_fn["sentence_categorical_accuracy"].update_state(
            tf.expand_dims(sentence_classes, -1),
            tf.nn.softmax(sentence_logits, axis=-1),
        )
        loss = attention_loss + token_arcface_loss + sentence_arcface_loss
        results = {}
        results["loss"] = loss
        results["attention_loss"] = attention_loss
        results["token_arcface_loss"] = token_arcface_loss
        results["sentence_arcface_loss"] = sentence_arcface_loss
        results["token_arcface_accuracy"] = self.metric_fn[
            "token_categorical_accuracy"
        ].result()
        results["sentence_arcface_accuracy"] = self.metric_fn[
            "sentence_categorical_accuracy"
        ].result()
        return results

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.metric_fn[k] for k in self.metric_fn.keys()]
