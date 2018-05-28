from keras import backend as K
from keras.layers import Conv3D, Dropout, Input, Conv3DTranspose, Flatten, Dense, concatenate
from keras.layers import Activation, Reshape, Permute, Lambda
from keras.models import Model


def dsc_loss(y_true, y_pred):
    dsc_class = K.sum(y_true * y_pred, axis=0) / (K.sum(y_true, axis=0) + K.sum(y_pred, axis=0))
    return 1 - 2 * dsc_class[0]


def get_brats_unet(input_shape, filters_list, kernel_size_list, nlabels, drop=0.2):
    inputs = Input(shape=input_shape, name='seg_inputs')

    curr_tensor = inputs
    conv_list = list()
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        conv = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        curr_tensor = Dropout(drop)(conv(curr_tensor))
        conv_list.append(curr_tensor)

    deconv_zip = zip(filters_list[-2::-1], kernel_size_list[-2::-1], conv_list[-2::-1])
    curr_tensor = Conv3DTranspose(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(curr_tensor)
    for i, (filters, kernel_size, prev_tensor) in enumerate(deconv_zip):
        concat = concatenate([prev_tensor, curr_tensor], axis=1)
        deconv = Conv3DTranspose(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        curr_tensor = Dropout(drop)(deconv(concat))

    dense = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first')
    curr_tensor = dense(curr_tensor)

    curr_tensor = Reshape((nlabels, -1))(curr_tensor)
    curr_tensor = Permute((2, 1))(curr_tensor)
    outputs = Activation('softmax', name='seg')(curr_tensor)

    net = Model(inputs=inputs, outputs=outputs)

    net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return net


def get_brats_roinet(input_shape, filters_list, kernel_size_list, nlabels = 2, drop=0.2):
    # Input
    inputs = Input(shape=input_shape, name='seg_inputs')

    ''' Convolutional part '''
    curr_tensor = inputs
    conv_list = list()
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        conv = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        curr_tensor = Dropout(drop)(conv(curr_tensor))
        conv_list.append(curr_tensor)

    ''' Deconvolutional part '''
    deconv_zip = zip(filters_list[-2::-1], kernel_size_list[-2::-1], conv_list[-2::-1])
    curr_tensor = Conv3DTranspose(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(curr_tensor)
    for i, (filters, kernel_size, prev_tensor) in enumerate(deconv_zip):
        concat = concatenate([prev_tensor, curr_tensor], axis=1)
        deconv = Conv3DTranspose(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        curr_tensor = Dropout(drop)(deconv(concat))

    dense = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first')
    curr_tensor = dense(curr_tensor)

    ''' Is there a tumor here? '''
    flat_tensor = Flatten()(curr_tensor)
    roi_out = Dense(nlabels, activation='softmax', name='localizer')(flat_tensor)

    ''' Tumor segmentation'''
    curr_tensor = Reshape((nlabels, -1))(curr_tensor)
    curr_tensor = Permute((2, 1))(curr_tensor)
    # Weight the output according to whether or not there is a tumor on that patch.
    lambda_mult = Lambda(lambda (x, y): x * y[-1])
    curr_tensor = lambda_mult ([curr_tensor, roi_out])
    block_out = Activation('softmax', name='seg')(curr_tensor)
    outputs = [roi_out, block_out]

    net = Model(inputs=inputs, outputs=outputs)

    net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return net


def get_brats_invunet(input_shape, filters_list, kernel_size_list, nlabels, drop=0.2):
    inputs = Input(shape=input_shape, name='seg_inputs')

    curr_tensor = inputs
    conv_list = list()
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        conv = Conv3DTranspose(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        curr_tensor = Dropout(drop)(conv(curr_tensor))
        conv_list.append(curr_tensor)

    deconv_zip = zip(filters_list[-2::-1], kernel_size_list[-2::-1], conv_list[-2::-1])
    curr_tensor = Conv3D(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(curr_tensor)
    for i, (filters, kernel_size, prev_tensor) in enumerate(deconv_zip):
        concat = concatenate([prev_tensor, curr_tensor], axis=1)
        deconv = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        curr_tensor = Dropout(drop)(deconv(concat))

    dense = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first')
    curr_tensor = dense(curr_tensor)

    curr_tensor = Reshape((nlabels, -1))(curr_tensor)
    curr_tensor = Permute((2, 1))(curr_tensor)
    outputs = Activation('softmax', name='seg')(curr_tensor)

    net = Model(inputs=inputs, outputs=outputs)

    net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return net


def get_brats_cnn(n_channels, filters_list, kernel_size_list, nlabels, dense_size, drop=0.2):
    # Init
    n_blocks = len(filters_list)
    input_shape_d = (n_channels, 3, 3, 3)
    input_shape_c = (n_channels,) + (n_blocks * 2 + 3,) * 3

    inputs_d = Input(shape=input_shape_d, name='d_inputs')
    inputs_c = Input(shape=input_shape_c, name='c_inputs')
    inputs = [inputs_d, inputs_c]

    tensor_d = inputs_d
    tensor_c = inputs_c
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        deconv = Conv3DTranspose(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        conv = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        tensor_d = Dropout(drop)(deconv(tensor_d))
        tensor_c = Dropout(drop)(conv(tensor_c))

    tensor_d = Conv3D(
        dense_size,
        kernel_size=tuple(map(lambda x: x-1, K.int_shape(tensor_d)[2:])),
        activation='relu',
        data_format='channels_first',
        name='d_dense'
    )(tensor_d)
    tensor_c = Conv3D(
        dense_size,
        kernel_size=tuple(map(lambda x: x-1, K.int_shape(tensor_c)[2:])),
        activation='relu',
        data_format='channels_first',
        name='c_dense'
    )(tensor_c)
    tensors = concatenate([Flatten()(tensor_d), Flatten()(tensor_c)])
    outputs = Dense(nlabels)(tensors)

    net = Model(inputs=inputs, outputs=outputs)

    net.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return net
