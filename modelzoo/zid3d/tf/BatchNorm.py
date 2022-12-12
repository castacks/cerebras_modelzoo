
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer
# from modelzoo.zid3d.tf import utils_smart_cond as smart_cont_tools

class BatchNormalizationLayer(BaseLayer):

    _USE_V2_BEHAVIOR = True

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
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
            boundary_casting, tf_summary, name=name, **kwargs
        )

        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.fused = False

        self._trainable_var = None
        self.trainable = trainable

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        if self._trainable_var is not None:
            self._trainable_var.update_value(value)

    def _get_trainable_var(self):
        if self._trainable_var is None:
            self._trainable_var = K.freezable_variable(
                    self._trainable, name=self.name + '_trainable')
        return self._trainable_var

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

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                                input_shape)
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [axis_to_dim[i] if i in axis_to_dim
                                         else 1 for i in range(ndims)]

        if self.scale:
            self.gamma = self.add_weight(
                    name='gamma',
                    shape=param_shape,
                    dtype=self._param_dtype,
                    initializer=self.gamma_initializer,
                    regularizer=self.gamma_regularizer,
                    constraint=self.gamma_constraint,
                    trainable=True,
                    experimental_autocast=False)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                    name='beta',
                    shape=param_shape,
                    dtype=self._param_dtype,
                    initializer=self.beta_initializer,
                    regularizer=self.beta_regularizer,
                    constraint=self.beta_constraint,
                    trainable=True,
                    experimental_autocast=False)
        else:
            self.beta = None

        try:
            # Disable variable partitioning when creating the moving mean and variance
            if hasattr(self, '_scope') and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_weight(
                    name='moving_mean',
                    shape=param_shape,
                    dtype=self._param_dtype,
                    initializer=self.moving_mean_initializer,
                    synchronization=tf_variables.VariableSynchronization.ON_READ,
                    trainable=False,
                    aggregation=tf_variables.VariableAggregation.MEAN,
                    experimental_autocast=False)

            self.moving_variance = self.add_weight(
                    name='moving_variance',
                    shape=param_shape,
                    dtype=self._param_dtype,
                    initializer=self.moving_variance_initializer,
                    synchronization=tf_variables.VariableSynchronization.ON_READ,
                    trainable=False,
                    aggregation=tf_variables.VariableAggregation.MEAN,
                    experimental_autocast=False)
        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor_v2(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (
                        variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                    K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def _assign_new_value(self, variable, value):
        with K.name_scope('AssignNewValue') as scope:
            with ops.colocate_with(variable):
                return state_ops.assign(variable, value, name=scope)

    def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
        return nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

    def _moments(self, inputs, reduction_axes, keep_dims):
        mean, variance = self._calculate_mean_and_var(inputs, reduction_axes, keep_dims)
        return mean, variance

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        if self._USE_V2_BEHAVIOR:
            if isinstance(training, int):
                training = bool(training)
            if base_layer_utils.is_in_keras_graph():
                training = math_ops.logical_and(training, self._get_trainable_var())
            elif not self.trainable:
                # When the layer is not trainable, it overrides the value passed from
                # model.
                training = self.trainable
        return training

    def layer_call(self, inputs, training=None):
        training = self._get_training_value(training)

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

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            mean, variance = self.moving_mean, self.moving_variance
        else:
            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = len(self.axis) > 1
            mean, variance = self._moments(
                    math_ops.cast(inputs, self._param_dtype),
                    reduction_axes,
                    keep_dims=keep_dims)

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = tf_utils.smart_cond(
                    training,
                    lambda: mean,
                    lambda: ops.convert_to_tensor_v2(moving_mean))
            variance = tf_utils.smart_cond(
                        training,
                        lambda: variance,
                        lambda: ops.convert_to_tensor_v2(moving_variance))

            new_mean, new_variance = mean, variance

            input_batch_size = None

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum, input_batch_size)

            def mean_update():
                true_branch = lambda: _do_update(self.moving_mean, new_mean)
                false_branch = lambda: self.moving_mean
                return tf_utils.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                true_branch = lambda: _do_update(self.moving_variance, new_variance)
                false_branch = lambda: self.moving_variance
                return tf_utils.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        # mean, variance = self.moving_mean, self.moving_variance

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
