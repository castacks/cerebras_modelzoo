

from tensorflow.python.ops import nn

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer
# from modelzoo.zid3d.tf import utils_smart_cond as smart_cont_tools

class NaiveBatchNormalizationLayer(BaseLayer):

    def __init__(
        self,
        axis=1,
        epsilon=1e-3,
        beta_initializer='zeros',
        gamma_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        trainable=True,
        name=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super().__init__(
            boundary_casting, tf_summary, name=name, trainable=trainable, **kwargs
        )

        self.epsilon = epsilon

        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.axis = [axis]

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            return dtypes.float32
        else:
            return self.dtype or dtypes.float32

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}

        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        param_shape = (list(axis_to_dim.values())[0],)

        self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)

        self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)

        self.built = True

    def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
        return nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

    def _moments(self, inputs, reduction_axes, keep_dims):
        mean, variance = self._calculate_mean_and_var(inputs, reduction_axes, keep_dims)
        return mean, variance

    def layer_call(self, inputs, training=None):

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        keep_dims = len(self.axis) > 1
        mean, variance = self._moments(
                math_ops.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims)

        mean = math_ops.cast(mean, inputs.dtype)
        variance = math_ops.cast(variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
        if scale is not None:
            scale = math_ops.cast(scale, inputs.dtype)
        # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
        # math in float16 hurts validation accuracy of popular models like resnet.
        outputs = nn.batch_normalization(inputs,
                                        _broadcast(mean),
                                        _broadcast(variance),
                                        offset,
                                        scale,
                                        self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        return outputs

    def call(self, inputs, **kwargs):
        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        output = self.layer_call(inputs, **kwargs)

        if self.tf_summary:
            output = summary_layer(output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape
