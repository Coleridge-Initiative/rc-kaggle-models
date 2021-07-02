import math

import tensorflow as tf
import tensorflow.keras.backend as K


class ArcFace(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""

    def __init__(
        self,
        num_classes,
        margin=0.5,
        logist_scale=30,
        easy_margin=False,
        centers=1,
        **kwargs
    ):
        super(ArcFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
        self.easy_margin = easy_margin
        self.sub_center = True if centers > 1 else False
        self.centers = centers

    def build(self, input_shape):
        if self.sub_center:
            self.w = self.add_weight(
                "weights",
                shape=[int(input_shape[0][-1]), self.centers, self.num_classes],
                trainable=True,
                dtype=tf.float32,
            )
        else:
            self.w = self.add_weight(
                "weights",
                shape=[int(input_shape[0][-1]), self.num_classes],
                trainable=True,
                dtype=tf.float32,
            )
        self.cos_m = tf.identity(math.cos(self.margin), name="cos_m")
        self.sin_m = tf.identity(math.sin(self.margin), name="sin_m")
        self.th = tf.identity(math.cos(math.pi - self.margin), name="th")
        self.mm = tf.multiply(math.sin(math.pi - self.margin), self.margin, name="mm")

    def call(self, inputs):
        embds, labels = inputs
        embds_norm = tf.norm(embds, axis=-1, keepdims=True)
        embds_norm = tf.clip_by_value(embds_norm, 10.0, 110.0)
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name="normed_embd")
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name="normed_weights")

        if self.sub_center:
            normed_w_reshape = tf.reshape(
                normed_w, (normed_w.shape[0], -1)
            )  # [512, K * n_classes]
            cos_t_subcenter = tf.matmul(
                normed_embds, normed_w_reshape, name="cos_t"
            )  # [B, K * n_classes]
            cos_t_subcenter = tf.reshape(
                cos_t_subcenter, (-1, self.centers, self.num_classes)
            )  # [B, K, n_classes]
            cos_t = tf.reduce_max(cos_t_subcenter, axis=1)  # [B, n_classes]
        else:
            cos_t = tf.matmul(
                normed_embds, normed_w, name="cos_t"
            )  # [B, F]sdot(F, n_class) ->> [B, n_class]

        cos_t = K.clip(
            cos_t, -1.0 + K.epsilon(), 1.0 - K.epsilon()
        )  # for numerical stability
        sin_t = tf.sqrt(1.0 - cos_t ** 2, name="sin_t")

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name="cos_mt"
        )  # cos(t + m) = cos(t) * cos(m) - sin(t) * sin(m)

        if self.easy_margin:
            mask = tf.cast(cos_t > 0, cos_mt.dtype)
            cos_mt = mask * cos_mt + (1.0 - mask) * cos_t
        else:
            mask = tf.cast(cos_t > self.th, cos_mt.dtype)
            cos_mt = mask * cos_mt + (1.0 - mask) * (cos_t - self.mm)

        labels = tf.one_hot(labels, depth=2, dtype=cos_mt.dtype)
        labels = tf.cast(labels, cos_mt.dtype)

        logists = labels * cos_mt + (1 - labels) * cos_t
        logists = tf.multiply(logists, self.logist_scale, "arcface_logist")

        return logists

    def get_config(self):
        config = super(ArcFace, self).get_config()
        config.update({"num_classes": self.num_classes})
        config.update({"margin": self.margin})
        config.update({"logist_scale": self.logist_scale})
        config.update({"easy_margin": self.easy_margin})
        config.update({"centers": self.centers})
        return config
