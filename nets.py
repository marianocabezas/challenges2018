from keras import backend as K
from keras.layers import Conv3D, Dropout, Input, Conv3DTranspose, concatenate
from keras.layers import Activation, Reshape, Permute
from keras.models import Model


def dsc_loss(y_true, y_pred):
    dsc_class = K.sum(y_true * y_pred, axis=0) / (K.sum(y_true, axis=0) + K.sum(y_pred, axis=0))
    return 1 - 2 * dsc_class[0]


def get_brats_unet(input_shape, filters_list, kernel_size_list, nlabels, drop=0.5):
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
