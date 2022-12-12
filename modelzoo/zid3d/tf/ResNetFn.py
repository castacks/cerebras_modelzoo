
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers

from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.AddLayer import AddLayer
from modelzoo.common.tf.layers.Conv2DLayer import Conv2DLayer
from modelzoo.common.tf.layers.MaxPool2DLayer import MaxPool2DLayer

from modelzoo.zid3d.tf.NaiveBatchNorm import NaiveBatchNormalizationLayer

def ResNet( stack_fn,
            preact,
            use_bias,
            model_name='resnet',
            input_tensor=None,
            input_shape=None,
            activation='relu6',
            data_format='channels_first',
            boundary_casting=False,
            tf_summary=False,
            dtype=dtypes.float32,):
    
    assert input_tensor is not None or input_shape is not None, 'Either input_tensor or input_shape must be specified. '

    with tf.compat.v1.name_scope(model_name):
        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        bn_axis = 1
        
        # x = layers.ZeroPadding2D(
        #         padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
        x = Conv2DLayer(64, 7, strides=2, use_bias=use_bias, name='conv1_conv',
                        padding='same',
                        data_format=data_format,
                        boundary_casting=boundary_casting,
                        tf_summary=tf_summary, 
                        dtype=dtype)(img_input)

        if not preact:
            x = NaiveBatchNormalizationLayer(
                    axis=bn_axis, epsilon=1.001e-5, name='conv1_bn',
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary, 
                    dtype=dtype)(x)
            x = ActivationLayer(activation, name='conv1_relu',
                                boundary_casting=boundary_casting,
                                tf_summary=tf_summary, 
                                dtype=dtype)(x)

        # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = MaxPool2DLayer( 3, strides=2, name='pool1_pool',
                            data_format=data_format,
                            boundary_casting=boundary_casting,
                            tf_summary=tf_summary, 
                            dtype=dtype )(x)

        x = stack_fn(x,
                     activation=activation,
                     data_format=data_format,
                     boundary_casting=boundary_casting,
                     tf_summary=tf_summary, 
                     dtype=dtype)

        if preact:
            x = NaiveBatchNormalizationLayer(
                    axis=bn_axis, epsilon=1.001e-5, name='post_bn',
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary, 
                    dtype=dtype)(x)
            x = ActivationLayer(activation, name='post_relu',
                                boundary_casting=boundary_casting,
                                tf_summary=tf_summary, 
                                dtype=dtype)(x)

        return x

def block1( x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None,
            activation='relu6',
            data_format='channels_first',
            boundary_casting=False,
            tf_summary=False,
            dtype=dtypes.float32, ):
    """A residual block.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
        name: string, block label.

    Returns:
        Output tensor for the residual block.
    """
    bn_axis = 1

    if conv_shortcut:
        shortcut = Conv2DLayer(
                4 * filters, 1, strides=stride, name=name + '_0_conv',
                data_format=data_format,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary, 
                dtype=dtype)(x)
        shortcut = NaiveBatchNormalizationLayer(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn',
                boundary_casting=boundary_casting,
                tf_summary=tf_summary, 
                dtype=dtype)(shortcut)
    else:
        shortcut = x

    x = Conv2DLayer(filters, 1, strides=stride, name=name + '_1_conv',
                    data_format=data_format,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary, 
                    dtype=dtype)(x)
    x = NaiveBatchNormalizationLayer(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn',
            boundary_casting=boundary_casting,
            tf_summary=tf_summary, 
            dtype=dtype)(x)
    x = ActivationLayer('activation', name=name + '_1_relu',
                        boundary_casting=boundary_casting,
                        tf_summary=tf_summary, 
                        dtype=dtype)(x)

    x = Conv2DLayer(
            filters, kernel_size, padding='same', name=name + '_2_conv',
            data_format=data_format,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary, 
            dtype=dtype)(x)
    x = NaiveBatchNormalizationLayer(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn',
            boundary_casting=boundary_casting,
            tf_summary=tf_summary, 
            dtype=dtype)(x)
    x = ActivationLayer(activation, name=name + '_2_relu',
                        boundary_casting=boundary_casting,
                        tf_summary=tf_summary, 
                        dtype=dtype)(x)

    x = Conv2DLayer(4 * filters, 1, name=name + '_3_conv',
                    data_format=data_format,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary, 
                    dtype=dtype)(x)
    x = NaiveBatchNormalizationLayer(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn',
            boundary_casting=boundary_casting,
            tf_summary=tf_summary, 
            dtype=dtype)(x)

    x = AddLayer(name=name + '_add',
                 boundary_casting=boundary_casting,
                 tf_summary=tf_summary, 
                 dtype=dtype)([shortcut, x])
    x = ActivationLayer(activation, name=name + '_out',
                        boundary_casting=boundary_casting,
                        tf_summary=tf_summary, 
                        dtype=dtype)(x)
    return x

def stack1( x, filters, blocks, stride1=2, name=None,
            activation='relu6',
            data_format='channels_first',
            boundary_casting=False,
            tf_summary=False,
            dtype=dtypes.float32, ):
    """A set of stacked residual blocks.

    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1',
               activation=activation,
               data_format=data_format,
               boundary_casting=boundary_casting,
               tf_summary=tf_summary,
               dtype=dtype)
    for i in range(2, blocks + 1):
        x = block1( x, filters, conv_shortcut=False, name=name + '_block' + str(i),
                    activation=activation,
                    data_format=data_format,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=dtype )
    return x

def ResNet50(
        input_tensor=None,
        input_shape=None,
        activation='relu6',
        data_format='channels_first',
        boundary_casting=False,
        tf_summary=False,
        dtype=dtypes.float32, ):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x, 
                 activation='relu6',
                 data_format='channels_first',
                 boundary_casting=False,
                 tf_summary=False,
                 dtype=dtypes.float32):
        
        add_kwargs = dict(
            activation=activation,
            data_format=data_format,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype)
        
        # In put size is 1/4 of the original feature.
        
        x = stack1(x,  64, 3, stride1=1, name='conv2', **add_kwargs) # 1/4
        x = stack1(x, 128, 4, stride1=2, name='conv3', **add_kwargs) # 1/8
        x = stack1(x, 256, 6, stride1=2, name='conv4', **add_kwargs) # 1/16
        x = stack1(x, 512, 3, stride1=2, name='conv5', **add_kwargs) # 1/32
        return x

    return ResNet( stack_fn, False, True, 'resnet50', 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   activation=activation,
                   data_format=data_format,
                   boundary_casting=boundary_casting,
                   tf_summary=tf_summary,
                   dtype=dtype )
