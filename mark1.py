
import tensorflow as tf
from tensorflow import keras

from classControlOCR import clasMatOcr

def conv2d(inputs, f = 32, k = (3,3), s = 1, activation=None, padding = 'valid'):

    return tf.keras.layers.Conv2D(filters = f, kernel_size = k ,strides=(s, s),
                                  padding=padding,
                                  activation=activation,
                                  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(inputs)

def conv1d(inputs, f = 32, k = 3, s = 1, activation=None, padding = 'valid'):

    return tf.keras.layers.Conv1D(filters = f, kernel_size = k ,strides=s,
                                  padding=padding,
                                  activation=activation,
                                  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(inputs)                                  
    
def leaky_relu(inputs, alpha = 0.2):
    
    return tf.keras.layers.LeakyReLU()(inputs)

def dropout(inputs, keep_prob):

    return tf.keras.layers.Dropout(keep_prob)(inputs)

def Flatten(inputs):
    
    return tf.keras.layers.Flatten()(inputs)

def Dense(inputs, units = 1024, use_bias = True, activation = None):
    
    return tf.keras.layers.Dense(units,activation=activation,use_bias=True,)(inputs)

def batch_norm(inputs):
    
    return tf.keras.layers.BatchNormalization(axis=-1,
                                              momentum=0.99,
                                              epsilon=0.001,
                                              center=True,
                                              scale=True,
                                              beta_initializer='zeros',
                                              gamma_initializer='ones',
                                              moving_mean_initializer='zeros',
                                              moving_variance_initializer='ones')(inputs)

def dense_layer(input_, reduccion, agrandamiento):

    dl_1 = conv2d(inputs = input_, f = reduccion, k = (1,1), s = 1)
    dl_1 = conv2d(inputs = dl_1, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
    dl_1 = leaky_relu(tf.keras.layers.concatenate([input_, dl_1]))

    dl_2 = conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
    dl_2 = conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
    dl_1 = leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

    dl_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(dl_1)

    dl_2 = conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
    dl_2 = conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
    dl_1 = leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

    dl_2 = conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
    dl_2 = conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
    dl_1 = leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

    return dl_1

def decode(inputs, sequence_length):

    return tf.nn.ctc_greedy_decoder(inputs, sequence_length=sequence_length)#features['seq_lens'])

def lossFunction(yTrue, yPred):

    yTrue = tf.cast(yTrue, dtype = tf.int32)
    yTrue = tf.contrib.layers.dense_to_sparse(yTrue,eos_token=200)

    return tf.nn.ctc_loss(labels = yTrue,
                          inputs = yPred,
                          sequence_length = clasMatOcr.batch_size*[32],
                          time_major=False)


def mark1():

    x = tf.keras.Input(shape=(clasMatOcr.dim_fil,clasMatOcr.dim_col), name='input_layer')
    h_c1 = tf.keras.layers.Permute((2,1))(x)

    h_c1 = leaky_relu(batch_norm(conv1d(inputs = h_c1, f = 64, k = 5, s = 2, padding = 'same')))
    h_c1 = leaky_relu(batch_norm(conv1d(inputs = h_c1, f = 128, k = 3, s = 1, padding = 'same')))
    h_c1 = leaky_relu(batch_norm(conv1d(inputs = h_c1, f = 256, k = 3, s = 2, padding = 'same')))
    h_c1 = leaky_relu(batch_norm(conv1d(inputs = h_c1, f = 512, k = 3, s = 1, padding = 'same')))

    h_c1 = leaky_relu(batch_norm(conv1d(inputs = h_c1, f = 1024, k = 3, s = 1, padding = 'same')))

    h_c1 = tf.keras.layers.Dropout(0.5)(h_c1)
    
    h_c1 = tf.keras.layers.Dense(1024)(h_c1)
    h_c1 = tf.keras.layers.Dense(len(clasMatOcr.dict) + 1)(h_c1)

    return x, h_c1