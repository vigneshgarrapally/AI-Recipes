# import the necessary packages
import tensorflow as tf
import keras_efficientnet_v2 as efn
def build_efv2smodel(HEIGHT,WIDTH):
    pretrained = efn.EfficientNetV2S(pretrained="imagenet",num_classes=0,input_shape=[HEIGHT,WIDTH, 3])
    x = pretrained.output
    x = tf.keras.layers.Dense(512, activation = tf.keras.layers.LeakyReLU(alpha=0.2))(x)
    x =  tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation = tf.keras.layers.LeakyReLU(alpha=0.2))(x)
    model = tf.keras.Model(pretrained.input,x)
    return model
