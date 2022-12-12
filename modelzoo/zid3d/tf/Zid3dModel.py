# Author of the original Zid3d model: Bowen Li <bowenli2@andrew.cmu.edu>\
# Author of this version based on Cerebras TF library: Yaoyu Hu <yaoyuh@andrew.cmu.edu>

"""
Zid3d model to be used with TF Estimator
"""
import tensorflow as tf
from tensorflow.compat.v1.losses import Reduction
from tensorflow.python.keras.layers import Flatten, concatenate

from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.Conv2DLayer import Conv2DLayer
from modelzoo.common.tf.layers.Conv2DTransposeLayer import Conv2DTransposeLayer
from modelzoo.common.tf.layers.MaxPool2DLayer import MaxPool2DLayer
from modelzoo.common.tf.metrics.dice_coefficient import dice_coefficient_metric
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.zid3d.tf.utils import color_codes

# from modelzoo.zid3d.tf.BatchNorm import BatchNormalizationLayer
from modelzoo.zid3d.tf.NaiveBatchNorm import NaiveBatchNormalizationLayer
from modelzoo.zid3d.tf.ResNetFn import ResNet50

class Zid3dModel(TFBaseModel):
    """
    Zid3d model to be used with TF Estimator
    """

    def __init__(self, params):
        super(Zid3dModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        self.num_output_channels = 1

        mparams = params["model"]

        self.initializer = mparams["initializer"]
        self.initializer_params = mparams.get("initializer_params")
        if self.initializer_params:
            self.initializer = getattr(
                tf.compat.v1.keras.initializers, self.initializer
            )(**self.initializer_params)

        self.bias_initializer = mparams["bias_initializer"]
        self.bias_initializer_params = mparams.get("bias_initializer_params")
        if self.bias_initializer_params:
            self.bias_initializer = getattr(
                tf.compat.v1.keras.initializers, self.bias_initializer
            )(**self.bias_initializer_params)

        # CS util params for layers
        self.boundary_casting = mparams["boundary_casting"]
        self.tf_summary = mparams["tf_summary"]

        self.mixed_precision = mparams["mixed_precision"]
        
        self.data_format = mparams["data_format"]
        self.enable_bias = mparams["enable_bias"]

        # Model trainer
        self.trainer = Trainer(
            params=params["optimizer"],
            tf_summary=self.tf_summary,
            mixed_precision=self.mixed_precision,
        )

    def build_model(self, features, mode):
        x = features

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        with tf.compat.v1.name_scope('feat_ext'):
            # x = NaiveBatchNormalizationLayer(
            #     axis=1,
            #     name='batch_norm',
            #     tf_summary=self.tf_summary,
            #     dtype=self.policy,
            # )(x)
            
            x = ResNet50(input_tensor=x,
                         boundary_casting=self.boundary_casting,
                         tf_summary=self.tf_summary,
                         dtype=self.policy)
            
        ##### Output
        logits = Conv2DLayer(
            filters=self.num_output_channels,
            kernel_size=1,
            activation="linear",
            padding="same",
            name="output_conv",
            data_format=self.data_format,
            use_bias=self.enable_bias,
            kernel_initializer=self.initializer,
            bias_initializer=self.bias_initializer,
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.policy,
        )(x)

        return logits

    def build_total_loss(self, logits, features, labels, mode):
        # Get input image and corresponding gt mask.
        input_image = features
        reshaped_mask_image = labels

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Flatten the logits
        flatten = Flatten(
            dtype="float16" if self.mixed_precision else "float32"
        )
        reshaped_logits = flatten(logits)

        # Binary Cross-Entropy loss
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            reshaped_mask_image,
            reshaped_logits,
            loss_collection=None,
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
        )

        # loss = tf.keras.losses.MeanSquaredError(
        #     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        # )(
        #     labels,
        #     input_image,
        # )

        return loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.trainer.build_train_ops(total_loss)

    def build_eval_metric_ops(self, logits, labels, features):
        """
        Evaluation metrics
        """

        reshaped_mask_image = labels

        reshaped_mask_image = tf.cast(reshaped_mask_image, dtype=tf.int32)

        # Ensure channels are the last dimension for the rest of eval
        # metric calculations. Otherwise, need to do the rest of ops
        # according to the channels dimension
        if self.data_format == "channels_first":
            logits = tf.transpose(a=logits, perm=[0, 2, 3, 1])

        pred = tf.reshape(
            logits, [tf.shape(input=logits)[0], -1, self.num_output_channels],
        )

        if self.num_output_channels == 1:
            pred = tf.concat(
                [tf.ones(pred.shape, dtype=pred.dtype) - pred, pred], axis=-1
            )

        pred = tf.argmax(pred, axis=-1)

        # ignore void classes
        ignore_classes_tensor = tf.constant(
            False, shape=reshaped_mask_image.shape, dtype=tf.bool
        )
        for ignored_class in self.eval_ignore_classes:
            ignore_classes_tensor = tf.math.logical_or(
                ignore_classes_tensor,
                tf.math.equal(
                    reshaped_mask_image,
                    tf.constant(
                        ignored_class,
                        shape=reshaped_mask_image.shape,
                        dtype=tf.int32,
                    ),
                ),
            )

        weights = tf.where(
            ignore_classes_tensor,
            tf.zeros_like(reshaped_mask_image),
            tf.ones_like(reshaped_mask_image),
        )

        metrics_dict = dict()

        metrics_dict["eval/accuracy"] = tf.compat.v1.metrics.accuracy(
            labels=reshaped_mask_image, predictions=pred, weights=weights,
        )

        return metrics_dict

    def get_evaluation_hooks(self, logits, labels, features):
        """ As a result of this TF issue, need to explicitly define summary
        hooks to able to log image summaries in eval mode
        https://github.com/tensorflow/tensorflow/issues/15332
        """
        return None
