import tensorflow as tf
from tensorflow.keras import layers, models

def bottleneck(input_tensor, filters, kernel, t, s, r=False):
    tchannel = tf.keras.backend.int_shape(input_tensor)[-1] * t
    x = layers.Conv2D(tchannel, kernel_size=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel, strides=s, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if r:
        x = layers.Add()([x, input_tensor])
    return x

def MobileNetV2(input_shape, k):
    input_tensor = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = bottleneck(x, 16, 3, t=1, s=1)
    x = bottleneck(x, 24, 3, t=6, s=2)
    x = bottleneck(x, 24, 3, t=6, s=1, r=True)
    x = bottleneck(x, 32, 3, t=6, s=2)
    x = bottleneck(x, 32, 3, t=6, s=1, r=True)
    x = bottleneck(x, 32, 3, t=6, s=1, r=True)
    x = bottleneck(x, 64, 3, t=6, s=2)
    x = bottleneck(x, 64, 3, t=6, s=1, r=True)
    x = bottleneck(x, 64, 3, t=6, s=1, r=True)
    x = bottleneck(x, 64, 3, t=6, s=1, r=True)
    x = bottleneck(x, 96, 3, t=6, s=1)
    x = bottleneck(x, 96, 3, t=6, s=1, r=True)
    x = bottleneck(x, 96, 3, t=6, s=1, r=True)
    x = bottleneck(x, 160, 3, t=6, s=2)
    x = bottleneck(x, 160, 3, t=6, s=1, r=True)
    x = bottleneck(x, 160, 3, t=6, s=1, r=True)
    x = bottleneck(x, 320, 3, t=6, s=1)

    x = layers.Conv2D(1280, kernel_size=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 1280))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(k, kernel_size=1)(x)
    x = layers.Flatten()(x)
    x = layers.Softmax()(x)

    model = models.Model(input_tensor, x)
    return model
