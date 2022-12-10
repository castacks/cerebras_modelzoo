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

from modelzoo.zid3d.tf.BatchNorm import BatchNormalizationLayer

class Zid3d(TFBaseModel):
    """
    Zid3d model to be used with TF Estimator
    """

    def __init__(self, params):
        super(Zid3d, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )
        
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
        
        self.tf_summary = mparams["tf_summary"]
        
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
            x = BatchNormalizationLayer(
                axis=1,
                name='batch_norm',
                tf_summary=self.tf_summary,
                dtype=self.dtype_policy,
            )(x)
        
        return x

    def build_total_loss(self, logits, features, labels, mode):
        input_image = features
        
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            labels,
            features,
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
        
        pred = features
        
        metrics_dict = dict()
        
        metrics_dict["eval/accuracy"] = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=pred,
        )
        
        return metrics_dict

    def get_evaluation_hooks(self, logits, labels, features):
        """ As a result of this TF issue, need to explicitly define summary
        hooks to able to log image summaries in eval mode
        https://github.com/tensorflow/tensorflow/issues/15332
        """
        return None
