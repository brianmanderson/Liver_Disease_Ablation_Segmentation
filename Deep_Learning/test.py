import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, save_model, load_model
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import os

def squeeze_first2axes_operator(x5d) :
    shape = tf.keras.backend.shape(x5d) # get dynamic tensor shape
    x5d = tf.keras.backend.reshape(x5d, [shape[0] * shape[1], shape[2], shape[3], tf.constant(1)])
    return x5d


def squeeze_first2axes_shape(x5d_shape):
    output_shape = (None, None, None, tf.constant(1))
    return output_shape

def return_og_shape(og):
    def break_up_operator(x4d):
        shape = tf.keras.backend.shape(og) # get dynamic tensor shape
        x5d = tf.keras.backend.reshape(x4d, [shape[0], shape[1], shape[2], shape[3], tf.constant(1)])
        return x5d
    return break_up_operator

def make_model():
    img = Input((None, None, None, 1), name='UNet_Input')
    image_2D = Lambda(squeeze_first2axes_operator, output_shape=(None, None, None, 1))(img)
    output = Lambda(return_og_shape(img), output_shape=img.shape)(image_2D)
    mm = Model(inputs=img, outputs=output)
    return mm

# validate model outputs
a = np.random.randn(5, 10, 15, 15, 1)
mm = make_model()
xx = mm.predict(tf.convert_to_tensor(a))
xxx = 1