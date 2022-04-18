import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training = False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def mish(layer):
    return layer * tf.math.tanh(tf.math.softplus(layer))

def convolution(input_layer, filter_shape, downsample = False, activate = True, 
                batchnorm = True, activate_type = 'mish'):

    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        _padding = 'valid'
        _strides = 2
    else:
        _padding = 'same'
        _strides = 1  
    

    
    conv = Conv2D(filters = filter_shape[-1], kernel_size = filter_shape[0],
                    strides = _strides, padding = _padding, use_bias = not batchnorm, kernel_regularizer = tf.keras.regularizers.l2(.0005),
                    kernel_initializer = tf.random_normal_initializer(stddev = .01),
                    bias_initializer= tf.constant_initializer(0.)
                    )(input_layer)
    
    if batchnorm: 
        conv = BatchNormalization()(conv)

    if activate:
        if activate_type == 'leaky':
            conv = tf.nn.leaky_relu(conv, alpha = .1)
        else:
            conv = mish(conv)
    return conv
        