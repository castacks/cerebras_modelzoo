import tensorflow as tf

from tensorflow.keras.mixed_precision.experimental import Policy

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers

from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.AddLayer import AddLayer
from modelzoo.common.tf.layers.Conv2DLayer import Conv2DLayer
from modelzoo.common.tf.layers.MaxPool2DLayer import MaxPool2DLayer

from modelzoo.zid3d.tf.NaiveBatchNorm import NaiveBatchNormalizationLayer

_DEFAULT_POLICY = Policy('mixed_float16', loss_scale=None)

def func_for_build(x,
                   model_name,
                   activation='relu',
                   data_format='channels_first',
                   boundary_casting=False,
                   tf_summary=False,
                   dtype=_DEFAULT_POLICY):
    # Additional kwargs for Conv2DLayer.
    kwargs_conv = dict(
        data_format=data_format,
        boundary_casting=boundary_casting,
        tf_summary=tf_summary,
        dtype=dtype
    )

    # Additional kwargs for batch normalization.
    kwargs_bn = dict(
        boundary_casting=boundary_casting,
        tf_summary=tf_summary,
        dtype=dtype
    )

    # Additional kwargs for activation layers.
    kwargs_act = dict(
        boundary_casting=boundary_casting,
        tf_summary=tf_summary,
        dtype=dtype
    )

    # with tf.compat.v1.name_scope(model_name):
    x = Conv2DLayer(64, 7,
                    strides=(1, 1),
                    padding='same',
                    use_bias=True,
                    name='conv0',
                    **kwargs_conv
                    )(x)

    return x
