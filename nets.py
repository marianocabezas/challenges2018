from keras import backend as K
from keras.layers import Conv2D, Conv3D, Conv3DTranspose, AveragePooling2D, Dropout, BatchNormalization
from keras.layers import Input, Activation, Reshape, Permute, Lambda, Flatten, Dense, concatenate
from keras.models import Model
from keras.applications.vgg16 import VGG16
from layers import ScalingLayer, ThresholdingLayer


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


def get_brats_roinet(input_shape, filters_list, kernel_size_list, nlabels=2, drop=0.2):
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


def get_brats_nets(n_channels, filters_list, kernel_size_list, nlabels, dense_size, drop=0.5):
    # Init
    n_blocks = len(filters_list)
    input_shape = (n_channels,) + (n_blocks * 2 + 3,) * 3

    inputs = Input(shape=input_shape, name='seg_inputs')

    cnn_tensor = inputs
    fcnn_tensor = inputs
    unet_tensor = inputs
    ucnn_tensor = inputs

    unet_list = list()
    ucnn_list = list()

    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_size_list)):
        conv_cnn = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        cnn_tensor = Dropout(drop)(conv_cnn(cnn_tensor))

        conv_fcnn = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        fcnn_tensor = Dropout(drop)(conv_fcnn(fcnn_tensor))

        conv_unet = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        unet_tensor = Dropout(drop)(conv_unet(unet_tensor))
        unet_list.append(unet_tensor)

        conv_ucnn = Conv3D(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        ucnn_tensor = Dropout(drop)(conv_ucnn(ucnn_tensor))
        ucnn_list.append(ucnn_tensor)

    # > Convolutional only stuff
    # CNN
    cnn_tensor = Dense(dense_size)(Flatten()(cnn_tensor))
    cnn_tensor = Dense(nlabels)(cnn_tensor)
    # FCNN
    fcnn_tensor = Conv3D(dense_size, kernel_size=(1, 1, 1), data_format='channels_first')(fcnn_tensor)
    fcnn_tensor = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first')(fcnn_tensor)
    fcnn_tensor = Reshape((nlabels, -1))(fcnn_tensor)
    fcnn_tensor = Permute((2, 1))(fcnn_tensor)

    # > U-stuff
    deconv_zip = zip(filters_list[-2::-1], kernel_size_list[-2::-1], unet_list[-2::-1], ucnn_list[-2::-1])
    unet_tensor = Conv3DTranspose(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(unet_tensor)
    ucnn_tensor = Conv3DTranspose(
        filters_list[-1],
        kernel_size=kernel_size_list[-1],
        activation='relu',
        data_format='channels_first'
    )(ucnn_tensor)

    for i, (filters, kernel_size, prev_unet_tensor, prev_ucnn_tensor) in enumerate(deconv_zip):
        concat_unet = concatenate([prev_unet_tensor, unet_tensor], axis=1)
        deconv_unet = Conv3DTranspose(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        unet_tensor = Dropout(drop)(deconv_unet(concat_unet))

        concat_ucnn = concatenate([prev_ucnn_tensor, ucnn_tensor], axis=1)
        deconv_ucnn = Conv3DTranspose(
            filters,
            kernel_size=kernel_size,
            activation='relu',
            data_format='channels_first'
        )
        ucnn_tensor = Dropout(drop)(deconv_ucnn(concat_ucnn))

    dense_unet = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first')
    unet_tensor = dense_unet(unet_tensor)
    unet_tensor = Reshape((nlabels, -1))(unet_tensor)
    unet_tensor = Permute((2, 1))(unet_tensor)

    dense_ucnn = Conv3D(nlabels, kernel_size=(1, 1, 1), data_format='channels_first')
    ucnn_tensor = dense_ucnn(ucnn_tensor)
    ucnn_tensor = Dense(nlabels)(Flatten()(ucnn_tensor))

    unet_out = Activation('softmax', name='unet_seg')(unet_tensor)
    cnn_out = Activation('softmax', name='cnn_seg')(cnn_tensor)
    fcnn_out = Activation('softmax', name='fcnn_seg')(fcnn_tensor)
    ucnn_out = Activation('softmax', name='ucnn_seg')(ucnn_tensor)

    unet = Model(inputs=inputs, outputs=unet_out)
    unet.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    cnn = Model(inputs=inputs, outputs=cnn_out)
    cnn.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    fcnn = Model(inputs=inputs, outputs=fcnn_out)
    fcnn.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    ucnn = Model(inputs=inputs, outputs=ucnn_out)
    ucnn.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Instead of having the 4 independent networks trained independently, we also make them sahre wieghts
    # and update them with a global network.
    nets_outputs = [unet(inputs), cnn(inputs), fcnn(inputs), ucnn(inputs)]
    nets = Model(inputs=inputs, outputs=nets_outputs)
    nets.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        loss_weights=[2, 2, 2, 2]
    )
    return nets, unet, cnn, fcnn, ucnn


def get_brats_ensemble(n_channels, n_blocks, unet, cnn, fcnn, ucnn, nlabels):
    # Init
    input_shape = (n_channels,) + (n_blocks * 2 + 3,) * 3

    inputs = Input(shape=input_shape, name='seg_inputs')

    # Final ensemble mode. It's basically a weighted average, where the original networks are frozen.
    for l in unet.layers:
        l.trainable = False
    unet.trainable = False
    unet.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    for l in cnn.layers:
        l.trainable = False
    cnn.trainable = False
    cnn.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    for l in fcnn.layers:
        l.trainable = False
    fcnn.trainable = False
    fcnn.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    for l in ucnn.layers:
        l.trainable = False
    ucnn.trainable = False
    ucnn.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    ensemble_tensor = concatenate([
        Flatten()(unet(inputs)),
        cnn(inputs),
        Flatten()(fcnn(inputs)),
        ucnn(inputs)]
    )
    ensemble_tensor = Dense(nlabels)(ensemble_tensor)
    ensemble_out = Activation('softmax', name='unet_seg')(ensemble_tensor)

    ensemble = Model(inputs=inputs, outputs=ensemble_out)
    ensemble.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return ensemble


def get_brats_survival(thresholds=[300, 450],n_slices=20, n_features=4, dense_size=256, dropout=0.1):
    # Input (3D volume of X*X*S) + other features (age, tumor volumes and resection status?)
    # This volume should be split into S inputs that will be passed to S VGG models.
    vol_input = Input(shape=(224, 224, n_slices, 3), name='vol_input')
    slice_inputs = map(lambda i: Lambda(lambda l: l[:, :, :, i, :])(vol_input), range(n_slices))

    feature_input = Input(shape=(n_features, ), name='feat_input')
    inputs = [vol_input, feature_input]

    # VGG init
    base_model = VGG16(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # vgg_out = map(base_model, slice_inputs)
    vgg_out = map(BatchNormalization(), map(base_model, slice_inputs))

    # - Conv2D
    # vgg_fcc1 = Conv2D(512, (1, 1), activation='relu')
    # vgg_fccout1 = map(Dropout(dropout),  map(vgg_fcc1, vgg_out))

    vgg_fcc1 = ScalingLayer()
    vgg_fccout1 = map(Dropout(dropout), map(Activation('relu'), map(vgg_fcc1, vgg_out)))

    vgg_pool1 = AveragePooling2D(2)
    vgg_poolout1 = map(vgg_pool1, vgg_fccout1)

    # - Scaling layer
    # vgg_fcc2 = ScalingLayer()
    # vgg_fccout2 = map(Dropout(dropout), map(Activation('relu'), map(vgg_fcc2, vgg_poolout1)))

    vgg_fcc2 = Conv2D(dense_size, (1, 1), activation='relu')
    vgg_fccout2 = map(BatchNormalization(), map(Dropout(dropout), map(vgg_fcc2, vgg_poolout1)))

    vgg_pool2 = AveragePooling2D(2)
    vgg_poolout2 = map(vgg_pool2, vgg_fccout2)

    # - Conv2D
    # vgg_fcc3 = Conv2D(dense_size, (1, 1), activation='relu')
    # vgg_fccout3 = map(Flatten(), map(Dropout(dropout), map(vgg_fcc3, vgg_poolout2)))

    vgg_fcc3 = ScalingLayer()
    vgg_fccout3 = map(Dropout(dropout), map(Activation('relu'), map(vgg_fcc3, vgg_poolout2)))

    vgg_dense = Dense(dense_size, kernel_initializer='normal', activation='linear')
    vgg_out = concatenate(map(Dropout(dropout), map(Flatten(), map(vgg_dense, vgg_fccout3))))

    # Here we add the final layers to compute the survival value
    final_tensor = concatenate([feature_input, vgg_out])
    output = Dense(1, kernel_initializer='normal', activation='linear', name='survival')(final_tensor)
    output_cat = Activation('softmax')(ThresholdingLayer(thresholds=thresholds)(output))
    # output_cat = Dense(3, activation='softmax')(output)

    survival_net = Model(inputs=inputs, outputs=[output, output_cat])
    survival_net.compile(
        optimizer='adam',
        loss=['mean_squared_error', 'categorical_crossentropy'],
        metrics=['accuracy']
    )

    return survival_net
