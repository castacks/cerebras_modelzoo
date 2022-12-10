
from tensorflow.keras.layers import BatchNormalization

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer

class BatchNormalizationLayer(BaseLayer):
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        trainable=True,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super().__init__(
            boundary_casting, tf_summary, **kwargs
        )
        
        self.layer = BatchNormalization(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            trainable=trainable,
            name=self.name,
            dtype=self.dtype_policy,
        )
        
    def call(self, inputs, **kwargs):
        if self.boundary_casting:
            inputs = boundary_cast(inputs)
            
        output = self.layer(inputs)
        
        if self.tf_summary:
            output = summary_layer(output)
        
        return output
