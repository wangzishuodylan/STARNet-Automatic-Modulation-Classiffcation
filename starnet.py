import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from acnet_repvgg_dbb_block_utils import DBB
from keras import models, regularizers
from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import scipy.io as scio
import numpy as np
from numpy import array
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import backend as K
from keras import models
from keras.layers import *
# from complexnn.conv import ComplexConv1D
# from complexnn.bn import ComplexBatchNormalization
# from complexnn.dense import ComplexDense
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import random
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, concatenate, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM, Lambda, Concatenate, Activation, Flatten, Multiply, Add, Subtract, CuDNNGRU
from keras import backend as K
import tensorflow as tf
import pandas as pd

signal_len = 128
modulation_num = 11

data_format = 'channels_first'

concat_axis = 3


def get_amp_phase(data):
    X_train_cmplx = data[:, 0, :] + 1j * data[:, 1, :]
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(data[:, 1, :], data[:, 0, :]) / np.pi
    X_train_amp = np.reshape(X_train_amp, (-1, 1, signal_len))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, signal_len))
    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    X_train = np.transpose(np.array(X_train), (0, 2, 1))
    for i in range(X_train.shape[0]):
        X_train[i, :, 0] = X_train[i, :, 0] / np.linalg.norm(X_train[i, :, 0], 2)

    return X_train


def rotate_matrix(theta):
    m = np.zeros((2, 2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    print(m)
    return m


def Rotate_DA(x, y):
    [N, L, C] = np.shape(x)
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi / 2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3 * np.pi / 2))

    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA

def TestDataset(snr):
    x = np.load(f"test/x_snr={snr}.npy")
    y = np.load(f"test/y_snr={snr}.npy")
    y = to_categorical(y)

    return x, y


from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, concatenate, Reshape
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers import CuDNNLSTM, Lambda, Concatenate, BatchNormalization, Activation
from keras import backend as K


def squeeze_excite_block(input, ratio=128):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # print(channel_axis.shape)
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def cal1(x):
    y = tf.keras.backend.cos(x)
    return y


def cal2(x):
    y = tf.keras.backend.sin(x)
    return y


def STARNET(weights=None,
           X_train=[128, 2],
           classes=11,
           **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    encoder_inputs = tf.keras.Input(shape=[128, 2],
                                    name='encoder_inputs')
    encoder_inputs = BatchNormalization()(encoder_inputs)
    # print(encoder_inputs.shape)
    in_shp = [2, 128]
    # input1 = tf.transpose(encoder_inputs, (0, 2, 1))

    input1 = tf.transpose(encoder_inputs, (0, 2, 1))
    # print(input1.shape)
    xm = Reshape([2, 128, 1], input_shape=in_shp)(input1)
    xm = Conv2D(4, kernel_size=(2, 7), strides=2, padding='same', kernel_regularizer=regularizers.l2(0.001),
                activation="relu", name='conv00', kernel_initializer='glorot_normal', data_format='channels_last')(xm)

    xm0 = xm
    xm0 = Conv2D(8, kernel_size=(1, 1), strides=(1, 2), padding='same', name='conv11',
                 kernel_initializer='glorot_normal',
                 kernel_regularizer=regularizers.l2(0.001), data_format='channels_last')(xm0)
    xm = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001),
                activation="relu", name='conv0', kernel_initializer='glorot_normal', data_format='channels_last')(xm)
    xm1 = xm
    xm1 = BatchNormalization()(xm1)
    xm = SeparableConv2D(4, kernel_size=(1, 3), strides=1, padding='same', name='conv1',
                         kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001),
                         data_format='channels_last')(xm)
    xm = keras.layers.Add()([xm, xm1])
    # xm = concatenate([xm, xm1], axis=3, name='Concatenate11')
    xm = Activation('relu')(xm)
    xm = squeeze_excite_block(xm)
    avgpool = tf.reduce_mean(xm, axis=3, keepdims=True, name='_spatial_avgpool')
    maxpool = tf.reduce_max(xm, axis=3, keepdims=True, name='_spatial_maxpool')
    spatial = Concatenate(axis=3)([avgpool, maxpool])

    spatial = Conv2D(1, (1, 3), strides=1, padding='same', name='vv2')(spatial)
    spatial_out = Activation('sigmoid', name='3vd')(spatial)
    xm = tf.multiply(xm, spatial_out)
    xm = Conv2D(4, kernel_size=(1, 1), strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(0.001),
                activation="relu", name='conv2', kernel_initializer='glorot_normal', data_format='channels_last')(xm)
    xm2 = xm
    xm2 = BatchNormalization()(xm2)
    xm = SeparableConv2D(4, kernel_size=(1, 3), strides=1, padding='same', name='conv3',
                         kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001),
                         data_format='channels_last')(xm)
    # xm = keras.layers.Add()([xm, xm2])
    xm = concatenate([xm, xm2], axis=3, name='Concatenate12')
    xm = keras.layers.Add()([xm, xm0])

    xm3 = xm
    xm3 = Conv2D(16, kernel_size=(1, 1), strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(0.001),
                 activation="relu", name='conv4', kernel_initializer='glorot_normal', data_format='channels_last')(xm3)
    xm = Conv2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001),
                activation="relu", name='conv5', kernel_initializer='glorot_normal', data_format='channels_last')(xm)
    xm4 = xm
    xm4 = BatchNormalization()(xm4)
    xm = SeparableConv2D(8, kernel_size=(1, 3), strides=1, padding='same', name='conv6',
                         kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001),
                         data_format='channels_last')(xm)
    xm = keras.layers.Add()([xm, xm4])
    # xm = concatenate([xm, xm4], axis=3, name='Concatenate13')
    xm = Activation('relu')(xm)
    xm = squeeze_excite_block(xm)
    avgpool = tf.reduce_mean(xm, axis=3, keepdims=True, name='_spa1tial_avgpool')
    maxpool = tf.reduce_max(xm, axis=3, keepdims=True, name='_spa1tial_maxpool')
    spatial = Concatenate(axis=3)([avgpool, maxpool])

    spatial = Conv2D(1, (1, 3), strides=1, padding='same', name='12v')(spatial)
    spatial_out = Activation('sigmoid', name='13v')(spatial)
    # xm = keras.layers.Add()([xm, spatial_out])
    xm = tf.multiply(xm, spatial_out)
    xm = Conv2D(8, kernel_size=(1, 1), strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(0.001),
                activation="relu", name='conv7', kernel_initializer='glorot_normal', data_format='channels_last')(xm)
    xm5 = xm
    xm5 = BatchNormalization()(xm5)
    xm = SeparableConv2D(8, kernel_size=(1, 3), strides=1, padding='same', name='conv8',
                         kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001),
                         data_format='channels_last')(xm)
    # xm = keras.layers.Add()([xm, xm5])
    xm = concatenate([xm, xm5], axis=3, name='Concatenate14')
    xm = keras.layers.Add()([xm, xm3])

    xm6 = xm
    xm6 = Conv2D(32, kernel_size=(1, 1), strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(0.001),
                 activation="relu", name='conv20', kernel_initializer='glorot_normal', data_format='channels_last')(xm6)
    xm = Conv2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001),
                activation="relu", name='conv21', kernel_initializer='glorot_normal', data_format='channels_last')(xm)
    xm7 = xm
    xm7 = BatchNormalization()(xm7)
    xm = SeparableConv2D(8, kernel_size=(1, 3), strides=1, padding='same', name='conv22',
                         kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001),
                         data_format='channels_last')(xm)
    xm = keras.layers.Add()([xm, xm7])
    # xm = concatenate([xm, xm7], axis=3, name='Concatenate15')
    xm = Activation('relu')(xm)
    xm = squeeze_excite_block(xm)
    avgpool = tf.reduce_mean(xm, axis=3, keepdims=True, name='_spatia2l_avgpool')
    maxpool = tf.reduce_max(xm, axis=3, keepdims=True, name='_spatia2l_maxpool')
    spatial = Concatenate(axis=3)([avgpool, maxpool])

    spatial = Conv2D(1, (1, 3), strides=1, padding='same', name='2v')(spatial)
    spatial_out = Activation('sigmoid', name='23v')(spatial)
    xm = tf.multiply(xm, spatial_out)
    xm = Conv2D(16, kernel_size=(1, 1), strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(0.001),
                activation="relu", name='conv23', kernel_initializer='glorot_normal', data_format='channels_last')(xm)
    xm8 = xm
    xm8 = BatchNormalization()(xm8)
    xm = SeparableConv2D(16, kernel_size=(1, 3), strides=1, padding='same', name='conv24',
                         kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.001),
                         data_format='channels_last')(xm)
    # xm = keras.layers.Add()([xm, xm8])
    xm = concatenate([xm, xm8], axis=3, name='Concatenate16')
    xm = keras.layers.Add()([xm, xm6])

    xm = keras.layers.GlobalAveragePooling2D()(xm)

    encoder_1, state_h_1 = tf.compat.v1.keras.layers.CuDNNGRU(units=32,
                                                              return_sequences=True,
                                                              return_state=True,
                                                              name='encoder_1')(encoder_inputs)

    drop_prob = 0
    drop_1 = tf.keras.layers.Dropout(drop_prob, name='drop_1')(encoder_1)
    encoder_2, state_h_2 = tf.compat.v1.keras.layers.CuDNNGRU(units=32,
                                                              return_state=True,
                                                              return_sequences=True,
                                                              name='encoder_2')(drop_1)

    # decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2),
    #                                           name='decoder')(encoder_2)
    state_h_2 = concatenate([0.7*state_h_2, 0.3*xm], axis=1, name='Concatenate1')
    xm = Reshape([64, 1, 1], input_shape=state_h_2.shape)(state_h_2)

    # 通道+空间注意力
    # xm = squeeze_excite_block(xm)
    # # xm = Reshape([64, 1, 1], input_shape=state_h_2.shape)(xm)
    # avgpool = tf.reduce_mean(xm, axis=3, keepdims=True, name='_spatial_avgpool')
    # maxpool = tf.reduce_max(xm, axis=3, keepdims=True, name='_spatial_maxpool')
    # spatial = Concatenate(axis=3)([avgpool, maxpool])
    #
    # spatial = Conv2D(64, (1, 1), strides=1, padding='same', name='2')(spatial)
    # spatial_out = Activation('sigmoid', name='3')(spatial)
    # xm = tf.multiply(xm, spatial_out)
    # print(xm.shape)
    # xm = tf.transpose(xm, (0, 3, 2, 1))

    # 3 Dense layers for classification with bn
    clf_dropout = 0.01
    xmm1 = Conv2D(4, kernel_size=1, strides=1, padding='same',
                           name='conv1d21', kernel_initializer='glorot_normal', data_format='channels_last')(xm)
    # xmm2 = tf.transpose(xmm1, (0, 3, 2, 1))
    # xm = keras.layers.GlobalAveragePooling2D()(xm)
    xmm1 = Flatten()(xmm1)
    xmm1 = Reshape([128, 2], input_shape=xm.shape)(xmm1)
    xm = tf.keras.layers.BatchNormalization(name='b1')(state_h_2)

    clf_dense_1 = tf.keras.layers.Dense(units=32,
                                        activation=tf.nn.relu,
                                        name='clf_dense_1')(xm)

    bn_1 = tf.keras.layers.BatchNormalization(name='bn_1')(clf_dense_1)

    clf_drop_1 = tf.keras.layers.Dropout(clf_dropout, name='clf_drop_1')(bn_1)

    clf_dense_2 = tf.keras.layers.Dense(units=16,
                                        activation=tf.nn.relu,
                                        name='clf_dense_2')(clf_drop_1)

    bn_2 = tf.keras.layers.BatchNormalization(name='bn_2')(clf_dense_2)

    clf_drop_2 = tf.keras.layers.Dropout(clf_dropout, name='clf_drop_2')(bn_2)

    clf_dense_3 = tf.keras.layers.Dense(units=modulation_num,
                                        name='clf_dense_3')(clf_drop_2)

    softmax = tf.keras.layers.Softmax(name='softmax')(clf_dense_3)

    model = tf.keras.Model(inputs=encoder_inputs, outputs=[xmm1, softmax])

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


# import keras
from keras.optimizers import adam_v2
# if __name__ == '__main__':
#     # for the RaioML2016.10a dataset
#     model = MCLDNN(classes=11)
#
#     # for the RadioML2016.10b dataset
#     # model = MCLDNN(classes=10)
#
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#     model.summary()