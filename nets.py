from keras import backend as K
from keras.layers import Dense, Conv3D, Dropout, Input, Conv3DTranspose, concatenate
from keras.layers import BatchNormalization, Activation, PReLU, Reshape, Permute
from keras.models import Model
from itertools import product
import numpy as np


def dsc_loss(y_true, y_pred):
    dsc_class = K.sum(y_true * y_pred, axis=0) / (K.sum(y_true, axis=0) + K.sum(y_pred, axis=0))
    return 1 - 2 * dsc_class[0]


def compile_network(inputs, outputs, weights):
    net = Model(inputs=inputs, outputs=outputs)

    net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        loss_weights=weights,
        metrics=['accuracy']
    )

    return net


def get_convolutional_block(input_l, filters_list, kernel_size_list, activation=PReLU, drop=0.5):
    for filters, kernel_size in zip(filters_list, kernel_size_list):
        input_l = Conv3D(filters, kernel_size=kernel_size, data_format='channels_first')(input_l)
        # input_l = BatchNormalization(axis=1)(input_l)
        input_l = activation()(input_l)
        input_l = Dropout(drop)(input_l)

    return input_l


def get_deconvolutional_block(input_l, filters_list, kernel_size_list, activation=PReLU, drop=0.5):
    for filters, kernel_size in zip(filters_list, kernel_size_list):
        input_l = Conv3DTranspose(filters, kernel_size=kernel_size, data_format='channels_first')(input_l)
        # input_l = BatchNormalization(axis=1)(input_l)
        input_l = activation()(input_l)
        input_l = Dropout(drop)(input_l)

    return input_l


def get_tissue_binary_stuff(input_l):
    csf = Dense(2)(input_l)
    gm = Dense(2)(input_l)
    wm = Dense(2)(input_l)
    csf_out = Activation('softmax', name='csf')(csf)
    gm_out = Activation('softmax', name='gm')(gm)
    wm_out = Activation('softmax', name='wm')(wm)

    return csf, gm, wm, csf_out, gm_out, wm_out


def get_brats_unet(input_shape, filters_list, kernel_size_list, nlabels, drop=0.5):
    s_inputs = Input(shape=input_shape, name='seg_inputs')

    conv_s = s_inputs
    conv_list = list()
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        conv = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        conv_s = BatchNormalization(axis=1)(Dropout(drop)(conv(conv_s)))
        conv_list.append(conv_s)

    deconv_zip = zip(filters_list[-2::-1], kernel_size_list[-2::-1], conv_list[-2::-1])
    conv_s = Conv3DTranspose(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(conv_s)
    for i, (filters, kernel_size, conv) in enumerate(deconv_zip):
        concat = concatenate([conv, conv_s], axis=0)
        deconv = Conv3DTranspose(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        conv_s = BatchNormalization(axis=1)(Dropout(drop)(deconv(concat)))

    full = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first', name='fc')
    full_s = full(conv_s)
    full_s = BatchNormalization(axis=1)(full_s)

    full_s = Reshape((nlabels, -1))(full_s)
    print(K.int_shape(full_s))
    full_s = Permute((2, 1))(full_s)
    seg = Activation('softmax', name='seg')(full_s)

    seg_net = Model(inputs=s_inputs, outputs=seg)

    seg_net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return seg_net
