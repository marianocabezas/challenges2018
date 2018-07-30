from __future__ import print_function
import argparse
import os
import csv
from time import strftime
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from nibabel import load as load_nii
from utils import color_codes, get_biggest_region
from data_creation import get_mask_centers, get_bounding_centers, get_mask_blocks
from data_creation import get_patch_labels, get_data, get_labels, load_images, get_reshaped_data
from data_manipulation.metrics import dsc_seg
from nets import get_brats_unet, get_brats_invunet, get_brats_ensemble, get_brats_nets, get_brats_survival
from keras import backend as K
from keras.applications.resnet50 import preprocess_input


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-t', '--train',
        dest='train_dir', default=None, nargs=2,
        help='Option to train a network. The second parameter is the folder with all the patients.'
    )
    group.add_argument(
        '-l', '--leave-one-out',
        dest='loo_dir', default='/home/mariano/DATA/Brats18TrainingData',
        help='Option to use leave-one-out. The second parameter is the folder with all the patients.'
    )

    # Network parameters
    parser.add_argument(
        '-i', '--patch-width',
        dest='patch_width', type=int, default=21,
        help='Initial patch size'
    )
    parser.add_argument(
        '-k', '--kernel-size',
        dest='conv_width', nargs='+', type=int, default=3,
        help='Size of all the convolutional kernels'
    )
    parser.add_argument(
        '-c', '--conv-blocks',
        dest='conv_blocks', type=int, default=5,
        help='Number of convolutional layers'
    )
    parser.add_argument(
        '-C', '--conv-blocks-seg',
        dest='conv_blocks_seg', type=int, default=3,
        help='Number of convolutional layers'
    )
    parser.add_argument(
        '-b', '--batch-size',
        dest='batch_size', type=int, default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '-B', '--batch-test-size',
        dest='test_size', type=int, default=32768,
        help='Batch size for testing'
    )
    parser.add_argument(
        '-s', '--down-sampling',
        dest='down_sampling', type=int, default=16,
        help='Downsampling ratio for the training data'
    )
    parser.add_argument(
        '-T', '--train-steps',
        dest='train_steps', type=int, default=10,
        help='Train steps for the curriculum learning of the survival network'
    )
    parser.add_argument(
        '-n', '--num-filters',
        action='store', dest='n_filters', nargs='+', type=int, default=[32],
        help='Number of filters for each convolutional'
    )
    parser.add_argument(
        '-d', '--dense-size',
        dest='dense_size', type=int, default=256,
        help='Number of units of the intermediate dense layer'
    )
    parser.add_argument(
        '-e', '--epochs',
        action='store', dest='epochs', type=int, default=10,
        help='Number of maximum epochs for training the segmentation task'
    )
    parser.add_argument(
        '-E', '--survival-epochs',
        action='store', dest='sepochs', type=int, default=50,
        help='Number of maximum epochs for training the survival task'
    )
    parser.add_argument(
        '-v', '--validation-rate',
        action='store', dest='val_rate', type=float, default=0.25,
        help='Rate of training samples used for validation for the segmentation task'
    )
    parser.add_argument(
        '-V', '--survival-validation-rate',
        action='store', dest='sval_rate', type=float, default=0.1,
        help='Rate of training samples used for validation for the survival task'
    )
    parser.add_argument(
        '-u', '--unbalanced',
        action='store_false', dest='balanced', default=True,
        help='Data balacing for training'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience', type=int, default=5,
        help='Maximum number of epochs without validation accuracy improvement for segmentation'
    )
    parser.add_argument(
        '-P', '--survival-patience',
        dest='spatience', type=int, default=10,
        help='Maximum number of epochs without validation accuracy improvement for survival'
    )
    parser.add_argument(
        '--no-flair',
        action='store_false', dest='use_flair', default=True,
        help='Don''t use FLAIR'
    )
    parser.add_argument(
        '--flair',
        action='store', dest='flair', default='_flair.nii.gz',
        help='FLAIR sufix name'
    )
    parser.add_argument(
        '--no-t1',
        action='store_false', dest='use_t1', default=True,
        help='Don''t use T1'
    )
    parser.add_argument(
        '--t1',
        action='store', dest='t1', default='_t1.nii.gz',
        help='T1 sufix name'
    )
    parser.add_argument(
        '--no-t1ce',
        action='store_false', dest='use_t1ce', default=True,
        help='Don''t use T1 with contrast'
    )
    parser.add_argument(
        '--t1ce',
        action='store', dest='t1ce', default='_t1ce.nii.gz',
        help='T1 with contrast enchancement sufix name'
    )
    parser.add_argument(
        '--no-t2',
        action='store_false', dest='use_t2', default=True,
        help='Don''t use T2'
    )
    parser.add_argument(
        '--t2',
        action='store', dest='t2', default='_t2.nii.gz',
        help='T2 sufix name'
    )
    parser.add_argument(
        '--labels',
        action='store', dest='labels', default='_seg.nii.gz',
        help='Labels image sufix'
    )
    parser.add_argument(
        '-L', '--n-labels',
        dest='nlabels', type=int, default=5,
        help='Number of labels (used to binarise classes)'
    )
    parser.add_argument(
        '-N', '--net',
        action='store', dest='netname', default='unet',
        help='Typor of network architecture'
    )
    parser.add_argument(
        '-S', '--number-slices',
        dest='n_slices', type=int, default=40,
        help='Initial patch size'
    )
    parser.add_argument(
        '-f', '--number-folds',
        dest='n_folds', type=int, default=5,
        help='Number of folds for the cross-validation'
    )

    networks = {
        'unet': get_brats_unet,
        'invunet': get_brats_invunet,
    }

    options = vars(parser.parse_args())
    options['net'] = networks[options['netname']]

    if options['netname'] is 'roinet':
        options['nlabels'] = 2

    return options


def get_patient_survival_features(path, p, p_features, test=False):
    # Init
    options = parse_inputs()
    roi_sufix = '_seg.nii.gz' if not test else '.nii.gz'
    roi = load_nii(os.path.join(path, p, p + roi_sufix)).get_data()
    brain = load_nii(os.path.join(path, p, p + options['t1'])).get_data()
    brain_vol = np.count_nonzero(brain)
    vol_features = map(lambda l: np.count_nonzero(roi == l) / brain_vol, [1, 2, 4])
    age_features = [float(p_features['Age']) / 100]
    features = [age_features + vol_features]

    return features


def get_patient_roi_slice(path, p):
    options = parse_inputs()
    n_slices = options['n_slices']

    # roi_sufix = '_seg.nii.gz' if not test else '.nii.gz'
    roi_sufix = '.nii.gz'
    roi = load_nii(os.path.join(path, p, p + roi_sufix)).get_data()
    brain = load_nii(os.path.join(path, p, p + options['t1'])).get_data()
    bounding_box_min = np.min(np.nonzero(brain), axis=1)
    bounding_box_max = np.max(np.nonzero(brain), axis=1)
    center_of_masses = np.mean(np.nonzero(roi), axis=1, dtype=np.int)
    slices = [[
        slice(bounding_box_min[0], bounding_box_max[0]),
        slice(bounding_box_min[1], bounding_box_max[1]),
        slice(center_of_masses[-1] - n_slices / 2, center_of_masses[-1] + n_slices / 2)
    ]]

    return slices


def get_survival_data(options, recession=list(['GTR', 'STR', 'NA']), test=False):
    # Init
    if options['train_dir'] is not None:
        path = options['train_dir'][0] if not test else options['train_dir'][1]
    else:
        path = options['loo_dir']

    with open(os.path.join(path, 'survival_data.csv')) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        names = csvreader.next()
        survivaldict = {
            p[0]: {
                field: value
                for field, value in zip(names[1:], p[1:])
            }
            for p in csvreader
        }

        flair_names = list()
        t1_names = list()
        t2_names = list()
        t1ce_names = list()
        survival = list()
        features = list()
        slices = list()
        names = list()
        for k, v in survivaldict.items():
            if v['ResectionStatus'] in recession:
                flair_names += [os.path.join(path, k, k + options['flair'])]
                t1_names += [os.path.join(path, k, k + options['t1'])]
                t1ce_names += [os.path.join(path, k, k + options['t1ce'])]
                t2_names += [os.path.join(path, k, k + options['t2'])]
                features += get_patient_survival_features(path, k, v, test)
                slices += get_patient_roi_slice(path, k)
                if not test:
                    survival += [float(v['Survival'])]
                else:
                    names += [k]

        image_names = np.stack([flair_names, t1_names, t1ce_names, t2_names], axis=1)

    features = np.array(features)

    if not test:
        survival = np.expand_dims(survival, axis=-1)
        packed_return = (image_names, survival, features, slices)
    else:
        packed_return = (names, image_names, features, slices)

    return packed_return


def get_names(sufix, path):
    options = parse_inputs()
    if path is None:
        path = options['train_dir'][0] if options['train_dir'] is not None else options['loo_dir']

    directories = filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])
    patients = sorted(directories)

    return map(lambda p: os.path.join(p, p.split('/')[-1] + sufix), patients)


def get_names_from_path(path=None):
    options = parse_inputs()
    # Prepare the names
    flair_names = get_names(options['flair'], path) if options['use_flair'] else None
    t2_names = get_names(options['t2'], path) if options['use_t2'] else None
    t1_names = get_names(options['t1'], path) if options['use_t1'] else None
    t1ce_names = get_names(options['t1ce'], path) if options['use_t1ce'] else None

    label_names = np.array(get_names(options['labels'], path))
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names


def check_dsc(gt_name, image, nlabels):
    gt_nii = load_nii(gt_name)
    gt = np.minimum(gt_nii.get_data(), nlabels-1).astype(dtype=np.uint8)
    labels = np.unique(gt.flatten())
    gt_nii.uncache()
    return [dsc_seg(gt == l, image == l) for l in labels[1:]]


def get_cnn_labels(centers, names, nlabels):
    y = get_labels(
        label_names=names,
        list_of_centers=centers,
        nlabels=nlabels,
        verbose=True
    )
    print('%s- Concatenating the labels (cnn)' % ' '.join([''] * 12))
    y = np.concatenate(y)
    return y


def get_fcnn_labels(centers, names, nlabels, patch_size=None):
    if patch_size is None:
        patch_size = (parse_inputs()['patch_width'],) * 3
    y = get_patch_labels(
        label_names=names,
        list_of_centers=centers,
        output_size=patch_size,
        nlabels=nlabels,
        verbose=True
    )
    print('%s- Concatenating the labels (fcnn)' % ' '.join([''] * 12))
    y = np.concatenate(y)
    return y


def get_cluster_labels(centers, names, nlabels):
    options = parse_inputs()
    conv_blocks = options['conv_blocks_seg']
    y_cnn = get_cnn_labels(centers, names, nlabels)
    y_fcnn = get_fcnn_labels(centers, names, nlabels, (3, 3, 3))
    y_unet = get_fcnn_labels(centers, names, nlabels, (conv_blocks * 2 + 3,) * 3)

    y = [y_unet, y_cnn, y_fcnn, y_cnn]

    return y


def train_survival_function(image_names, survival, features, slices, save_path, sufix=''):
    # Init
    options = parse_inputs()
    c = color_codes()
    # Prepare the net hyperparameters
    epochs = options['sepochs']
    n_slices = options['n_slices']

    ''' Net preparation '''
    net = get_brats_survival(n_slices=n_slices, n_features=features.shape[-1])
    net_name = os.path.join(save_path, 'brats2018-survival%s.mdl' % sufix)
    net.save(net_name)
    # checkpoint = 'brats2018-survival%s.hdf5' % sufix
    # net.summary()

    ''' Training '''
    train_steps = options['train_steps']
    try:
        net.load_weights(os.path.join(save_path, 'brats2018-survival%s.hdf5' % (train_steps - 1)))
        print(
            '%s[%s] %sSurvival network weights %sloaded%s' % (
                c['c'], strftime("%H:%M:%S"), c['g'],
                c['b'], c['nc']
            )
        )
    except IOError:
        trainable_params = int(np.sum([K.count_params(w) for w in set(net.trainable_weights)]))
        print(
            '%s[%s] %sTraining the survival network %s(%s%d %sparameters)' % (
                c['c'], strftime("%H:%M:%S"), c['g'], c['nc'],
                c['b'], trainable_params, c['nc']
            )
        )

        # Data preparation
        x_vol = get_reshaped_data(image_names, slices, (224, 224), n_slices=n_slices, verbose=True)
        print('%s- Concatenating the data' % ' '.join([''] * 12))
        x_vol = preprocess_input(np.stack(x_vol, axis=0))

        print('%s-- X (volume) shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, x_vol.shape))))
        print('%s-- X (features) shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, features.shape))))
        print('%s-- Y shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, survival.shape))))

        checkpoint = 'brats2018-survival%s-step.hdf5' % sufix

        try:
            # net.load_weights(os.path.join(save_path, checkpoint))
            net.load_weights(os.path.join(save_path, 'brats2018-survival%s-step%d.hdf5' % (sufix, train_steps - 1)))
            print(
                '%s[%s] %sSurvival network weights %sloaded%s' % (
                    c['c'], strftime("%H:%M:%S"), c['g'],
                    c['b'], c['nc']
                )
            )
        except IOError:

            ''' Normal version '''
            # callbacks = [
            #     EarlyStopping(
            #         monitor='val_loss',
            #         patience=options['spatience']
            #     ),
            #     ModelCheckpoint(
            #         os.path.join(save_path, checkpoint),
            #         monitor='val_loss',
            #         save_best_only=True
            #     )
            # ]

            callbacks = [
                EarlyStopping(
                    monitor='loss',
                    patience=options['spatience']
                ),
                ModelCheckpoint(
                    os.path.join(save_path, checkpoint),
                    monitor='loss',
                    save_best_only=True
                )
            ]

            print('%s- Randomising the training data' % ' '.join([''] * 12))
            idx = np.random.permutation(range(len(features)))

            x_vol = x_vol[idx].astype(np.float32)
            x_feat = features[idx].astype(np.float32)
            x = [x_vol, x_feat]

            y = survival[idx].astype(np.float32)
            y_cat = np.squeeze(np.stack([y < 300, np.logical_and(y >= 300, y < 450), y >= 450], axis=1))

            # net.fit(x, y, batch_size=8, validation_split=options['sval_rate'], epochs=epochs, callbacks=callbacks)
            net.fit(x, [y, y_cat], batch_size=8, epochs=epochs, callbacks=callbacks)
            net.load_weights(os.path.join(save_path, checkpoint))

            ''' Curriculum learning version '''
            # sorted_idx = np.squeeze(np.argsort(survival, axis=0))
            # for i in range(train_steps):
            #     checkpoint = 'brats2018-survival%s-step%d.hdf5' % (sufix, i)
            #     net = get_brats_survival(n_slices=n_slices, n_features=features.shape[-1])
            #     net_name = os.path.join(save_path, 'brats2018-survival%s.mdl' % sufix)
            #     net.save(net_name)
            #     try:
            #         net.load_weights(os.path.join(save_path, checkpoint))
            #         print(
            #             '%s[%s] %sSurvival network weights %sloaded%s' % (
            #                 c['c'], strftime("%H:%M:%S"), c['g'],
            #                 c['b'], c['nc']
            #             )
            #         )
            #     except IOError:
            #         print('%s- Selecting the next set of samples (step %d/%d)' %
            #               (' '.join([''] * 12), i + 1, train_steps))
            #         current_idx = np.concatenate([
            #             sorted_idx[:((i + 1) * len(sorted_idx)) / (2 * train_steps)],
            #             sorted_idx[-((i + 1) * len(sorted_idx)) / (2 * train_steps):],
            #         ])
            #
            #         callbacks = [
            #             EarlyStopping(
            #                 monitor='loss',
            #                 patience=options['spatience']
            #             ),
            #             ModelCheckpoint(
            #                 os.path.join(save_path, checkpoint),
            #                 monitor='loss',
            #                 save_best_only=True
            #             )
            #         ]
            #
            #         idx = np.random.permutation(range(len(current_idx)))
            #
            #         x = [x_vol[current_idx][idx].astype(np.float32), features[current_idx][idx].astype(np.float32)]
            #
            #         y = survival[current_idx][idx].astype(np.float32)
            #
            #         net.fit(x, y, batch_size=8, epochs=epochs, callbacks=callbacks)
            #         net.load_weights(os.path.join(save_path, checkpoint))

            ''' Average version '''
            # for i in range(train_steps):
            #     checkpoint = 'brats2018-survival%s-step%d.hdf5' % (sufix, i)
            #     try:
            #         net.load_weights(os.path.join(save_path, checkpoint))
            #         print(
            #             '%s[%s] %sSurvival network weights %sloaded%s' % (
            #                 c['c'], strftime("%H:%M:%S"), c['g'],
            #                 c['b'], c['nc']
            #             )
            #         )
            #     except IOError:
            #         callbacks = [
            #             EarlyStopping(
            #                 monitor='loss',
            #                 patience=options['spatience']
            #             ),
            #             ModelCheckpoint(
            #                 os.path.join(save_path, checkpoint),
            #                 monitor='loss',
            #                 save_best_only=True
            #             )
            #         ]
            #
            #         print('%s- Randomising the training data' % ' '.join([''] * 12))
            #         idx = np.random.permutation(range(len(features)))
            #
            #         x_vol = x_vol[idx].astype(np.float32)
            #         x_feat = features[idx].astype(np.float32)
            #         x = [x_vol, x_feat]
            #
            #         y = survival[idx].astype(np.float32)
            #
            #         net.fit(x, y, batch_size=8, epochs=epochs, callbacks=callbacks)
            #         net.load_weights(os.path.join(save_path, checkpoint))

    return net


def test_survival(net, image_names, features, slices, n_slices):
    c = color_codes()
    x_vol = get_reshaped_data(image_names, slices, (224, 224), n_slices=n_slices)
    x_vol = np.stack(x_vol, axis=0)

    x = [x_vol.astype(np.float32), np.array(features, dtype=np.float32)]
    print(
        '%s[%s] %sTesting the survival network %s' % (
            c['c'], strftime("%H:%M:%S"), c['g'], c['nc']
        )
    )
    survival = net.predict(x, batch_size=1)[0]
    return np.fabs(np.squeeze(survival))


def train_seg_function(image_names, label_names, brain_centers, save_path):
    # Init
    options = parse_inputs()
    c = color_codes()

    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters * conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks

    # Prepare the sufix that will be added to the results for the net and images
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    flair_s = '.noflair' if not options['use_flair'] else ''
    t1_s = '.not1' if not options['use_t1'] else ''
    t1ce_s = '.not1ce' if not options['use_t1ce'] else ''
    t2_s = '.not2' if not options['use_t2'] else ''
    images_s = flair_s + t1_s + t1ce_s + t2_s
    params_s = (options['netname'], images_s, options['nlabels'], patch_width, conv_s, filters_s, epochs)
    sufix = '.%s%s.l%d.p%d.c%s.n%s.e%d' % params_s

    '''Tumor ROI stuff'''
    # Training for the ROI
    input_shape = (image_names.shape[-1],) + patch_size
    net = options['net'](
        input_shape=input_shape,
        filters_list=filters_list,
        kernel_size_list=kernel_size_list,
        nlabels=options['nlabels']
    )
    train_seg(
        image_names=image_names,
        label_names=label_names,
        train_centers=brain_centers,
        net=net,
        save_path=save_path,
        sufix=sufix,
        nlabels=options['nlabels']
    )

    '''Tumor segmentation stuff'''
    # Training for the tumor inside the ROI
    # First we should retest each training image and get the test tumor mask
    # and join it with the GT mask. This new mask after dilation will give us
    # the training centers.
    print('%s%s<Creating the tumor masks for the training data>%s' % (
        ''.join([' '] * 14), c['g'], c['nc']
    ))
    masks = map(lambda labels: load_nii(labels).get_data().astype(np.bool), label_names)

    # > Ensemble training
    #
    # I should probably try a sliding window and the random sampling version to see which one is better.
    # Sadly, I have a feeling that the sliding window approach should be worse.
    print('%s- Extracting centers from the tumor ROI' % ' '.join([''] * 15))
    train_centers = get_mask_centers(masks)
    train_centers = map(
        lambda centers: map(
            tuple,
            np.random.permutation(centers)[::options['down_sampling']].tolist()
        ),
        train_centers
    )
    print('%s- %d centers will be used' % (' '.join([''] * 15), sum(map(len, train_centers))))

    dense_size = options['dense_size']
    conv_blocks_seg = options['conv_blocks_seg']
    nets, unet, cnn, fcnn, ucnn = get_brats_nets(
        n_channels=image_names.shape[-1],
        filters_list=n_filters * conv_blocks_seg,
        kernel_size_list=[conv_width] * conv_blocks_seg,
        nlabels=options['nlabels'],
        dense_size=dense_size
    )

    # First we train the nets inside the cluster net (I don't know what other name I could
    # give to that architecture).
    train_seg(
        image_names=image_names,
        label_names=label_names,
        train_centers=train_centers,
        net=nets,
        save_path=save_path,
        sufix='-nets-%s.d%d' % (sufix, dense_size),
        nlabels=options['nlabels'],
        net_type='nets'
    )

    # Then we train the Dense/Fully Connected layer that defines the ensemble.
    # The previous networks should be frozen here.
    ensemble = get_brats_ensemble(
        n_channels=image_names.shape[-1],
        n_blocks=conv_blocks_seg,
        unet=unet,
        cnn=cnn,
        fcnn=fcnn,
        ucnn=ucnn,
        nlabels=options['nlabels']
    )
    train_seg(
        image_names=image_names,
        label_names=label_names,
        train_centers=train_centers,
        net=ensemble,
        save_path=save_path,
        sufix='-ensemble-%s.d%d' % (sufix, dense_size),
        nlabels=options['nlabels'],
        net_type='ensemble'
    )

    return net, ensemble


def train_seg(net, image_names, label_names, train_centers, save_path, sufix, nlabels, net_type='unet'):
    options = parse_inputs()
    conv_blocks = options['conv_blocks_seg']
    patch_width = options['patch_width']
    patch_size = (patch_width,) * 3 if net_type == 'unet' else (conv_blocks * 2 + 3,) * 3
    c = color_codes()
    # Prepare the net hyperparameters
    epochs = options['epochs']
    batch_size = options['batch_size']

    net_name = os.path.join(save_path, 'brats2018%s.mdl' % sufix)
    net.save(net_name)

    checkpoint = 'brats2018%s.hdf5' % sufix
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=options['patience']
        ),
        ModelCheckpoint(
            os.path.join(save_path, checkpoint),
            monitor='val_loss',
            save_best_only=True
        )
    ]

    try:
        net.load_weights(os.path.join(save_path, checkpoint))
    except IOError:
        trainable_params = int(np.sum([K.count_params(w) for w in set(net.trainable_weights)]))
        print(
            '%s[%s] %sTraining the network %s%s %s(%s%d %sparameters)' % (
                c['c'], strftime("%H:%M:%S"), c['g'], c['b'], net_type, c['nc'],
                c['b'], trainable_params, c['nc']
            )
        )

        # net.summary()
        x = get_data(
            image_names=image_names,
            list_of_centers=train_centers,
            patch_size=patch_size,
            verbose=True,
        )
        print('%s- Concatenating the data' % ' '.join([''] * 12))
        x = np.concatenate(x)
        get_labels_dict = {
            'unet': lambda: get_fcnn_labels(train_centers, label_names, nlabels),
            'ensemble': lambda: get_cnn_labels(train_centers, label_names, nlabels),
            'nets': lambda: get_cluster_labels(train_centers, label_names, nlabels),
        }
        y = get_labels_dict[net_type]()
        print('%s-- Using %d blocks of data' % (
            ' '.join([''] * 12),
            len(x)
        ))
        print('%s-- X shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, x.shape))))
        if type(y) is not list:
            print('%s-- Y shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, y.shape))))
        else:
            y_message = '%s-- Y shape: (%s)'
            for yi in y:
                print(y_message % (' '.join([''] * 12), ', '.join(map(str, yi.shape))))

        print('%s- Randomising the training data' % ' '.join([''] * 12))
        idx = np.random.permutation(range(len(x)))

        x = x[idx]
        y = y[idx] if type(y) is not list else map(lambda yi: yi[idx], y)

        print('%s%sStarting the training process (%s%s%s%s) %s' % (
            ' '.join([''] * 12),
            c['g'],
            c['b'], net_type, c['nc'],
            c['g'], c['nc'])
              )
        net.fit(x, y, batch_size=batch_size, validation_split=options['val_rate'], epochs=epochs, callbacks=callbacks)
        net.load_weights(os.path.join(save_path, checkpoint))


def test_seg(net, p, outputname, nlabels, mask=None, verbose=True):

    c = color_codes()
    options = parse_inputs()
    p_name = p[0].rsplit('/')[-2]
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    outputname_path = os.path.join(patient_path, outputname + '.nii.gz')
    try:
        roi_nii = load_nii(outputname_path)
        if verbose:
            print('%s%s<%s%s%s%s - probability map loaded>%s' % (
                ''.join([' '] * 14), c['g'], c['b'], p_name, c['nc'], c['g'], c['nc']
            ))
    except IOError:
        roi_nii = load_nii(p[0])
        # Image loading
        if mask is None:
            # This is the unet path
            x = np.expand_dims(np.stack(load_images(p), axis=0), axis=0)
            # Network parameters
            conv_blocks = options['conv_blocks']
            n_filters = options['n_filters']
            filters_list = n_filters if len(n_filters) > 1 else n_filters * conv_blocks
            conv_width = options['conv_width']
            kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks

            image_net = options['net'](x.shape[1:], filters_list, kernel_size_list, nlabels)
            # We should copy the weights here (if not using roinet)
            for l_new, l_orig in zip(image_net.layers[1:], net.layers[1:]):
                l_new.set_weights(l_orig.get_weights())

            # Now we can test
            if verbose:
                print('%s[%s] %sTesting the network%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
            # Load only the patient images
            if verbose:
                print('%s%s<Creating the probability map for %s%s%s%s - %s%s%s %s>%s' % (
                    ''.join([' '] * 14),
                    c['g'],
                    c['b'], p_name, c['nc'], c['g'],
                    c['b'], outputname_path, c['nc'],
                    c['g'], c['nc']
                ))
            pr_maps = image_net.predict(x, batch_size=options['test_size'])
            image = np.argmax(pr_maps, axis=-1).reshape(x.shape[2:])
            image = get_biggest_region(image)
        else:
            # This is the ensemble path
            image = np.zeros_like(mask, dtype=np.int8)
            options = parse_inputs()
            conv_blocks = options['conv_blocks_seg']
            test_centers = get_mask_blocks(mask)
            x = get_data(
                image_names=[p],
                list_of_centers=[test_centers],
                patch_size=(conv_blocks * 2 + 3,) * 3,
                verbose=verbose,
            )
            if verbose:
                print('%s- Concatenating the data x' % ' '.join([''] * 12))
            x = np.concatenate(x)
            pr_maps = net.predict(x, batch_size=options['test_size'])
            [x, y, z] = np.stack(test_centers, axis=1)
            image[x, y, z] = np.argmax(pr_maps, axis=1).astype(dtype=np.int8)

        roi_nii.get_data()[:] = image
        roi_nii.to_filename(outputname_path)
    return roi_nii


def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters * conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks
    # Data loading parameters

    # Prepare the sufix that will be added to the results for the net and images
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    flair_s = '.noflair' if not options['use_flair'] else ''
    t1_s = '.not1' if not options['use_t1'] else ''
    t1ce_s = '.not1ce' if not options['use_t1ce'] else ''
    t2_s = '.not2' if not options['use_t2'] else ''
    images_s = flair_s + t1_s + t1ce_s + t2_s
    params_s = (options['netname'], images_s, options['nlabels'], patch_width, conv_s, filters_s, epochs)
    sufix = '.%s%s.l%d.p%d.c%s.n%s.e%d' % params_s
    train_data, _ = get_names_from_path()

    unet_seg_results = list()
    unet_roi_results = list()
    ensemble_seg_results = list()
    ensemble_roi_results = list()
    image_names, label_names = get_names_from_path()
    print('%s[%s] %s<BRATS 2018 pipeline testing>%s' % (c['c'], strftime("%H:%M:%S"), c['y'], c['nc']))
    print('%s[%s] %sCenter computation%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
    # Block center computation
    # overlap = 0 if options['netname'] != 'roinet' else patch_width / 4
    # brain_centers = get_bounding_centers(image_names, patch_width, overlap)
    test_dir = options['train_dir'][1] if options['train_dir'] is not None else None

    if test_dir is None:
        ''' <Segmentation task> '''
        #
        # print('%s[%s] %sStarting leave-one-out (segmentation)%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
        # for i in range(len(image_names)):
        #     ''' Training '''
        #     # Data separation
        #     p = image_names[i]
        #     train_images = np.asarray(image_names.tolist()[:i] + image_names.tolist()[i + 1:])
        #     train_labels = np.asarray(label_names.tolist()[:i] + label_names.tolist()[i + 1:])
        #     train_centers = brain_centers[:i] + brain_centers[i + 1:]
        #     # Patient stuff
        #     p_name = p[0].rsplit('/')[-2]
        #     patient_path = '/'.join(p[0].rsplit('/')[:-1])
        #
        #     # Data stuff
        #     print('%s[%s] %sPatient %s%s%s %s(%s%d%s%s/%d)%s' % (
        #         c['c'], strftime("%H:%M:%S"),
        #         c['g'], c['b'], p_name, c['nc'],
        #         c['c'], c['b'], i+1, c['nc'], c['c'], len(brain_centers), c['nc']
        #     ))
        #
        #     # > Training for everything
        #
        #     net, ensemble = train_seg_function(train_images, train_labels, train_centers, save_path=patient_path)
        #
        #     ''' Testing '''
        #     # > Testing for the tumor ROI
        #     #
        #     # We first test with the ROI segmentation net.
        #
        #     image_unet = test_seg(net, p, p_name, options['nlabels'])
        #     # image_unet = test_seg(net, p, p_name + '.unet.test' + sufix, options['nlabels'])
        #
        #     seg_dsc = check_dsc(label_names[i], image_unet.get_data(), options['nlabels'])
        #     roi_dsc = check_dsc(label_names[i], image_unet.get_data().astype(np.bool), 2)
        #
        #     dsc_string = c['g'] + '/'.join(['%f'] * len(seg_dsc)) + c['nc'] + ' (%f)'
        #     print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' Unet DSC: ' +
        #           dsc_string % tuple(seg_dsc + roi_dsc))
        #
        #     unet_seg_results.append(seg_dsc)
        #     unet_roi_results.append(roi_dsc)
        #
        #     # > Testing for the tumor inside the ROI
        #     #
        #     # All we need to do now is test with the ensemble.
        #     image_cnn = test_seg(
        #         ensemble,
        #         p,
        #         p_name,
        #         options['nlabels'],
        #         mask=image_unet.get_data().astype(np.bool),
        #         verbose=False
        #     )
        #
        #     seg_dsc = check_dsc(label_names[i], image_cnn.get_data(), options['nlabels'])
        #     roi_dsc = check_dsc(label_names[i], image_cnn.get_data().astype(np.bool), 2)
        #
        #     dsc_string = c['g'] + '/'.join(['%f'] * len(seg_dsc)) + c['nc'] + ' (%f)'
        #     print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' CNN DSC: ' +
        #           dsc_string % tuple(seg_dsc + roi_dsc))
        #
        #     ensemble_seg_results.append(seg_dsc)
        #     ensemble_roi_results.append(roi_dsc)
        #
        # unet_r_dsc = np.mean(unet_roi_results)
        # print('Final ROI results DSC: %f' % unet_r_dsc)
        # unet_f_dsc = tuple(map(
        #     lambda k: np.mean(
        #         [dsc[k] for dsc in unet_seg_results if len(dsc) > k]
        #     ),
        #     range(3)
        # ))
        # print('Final Unet results DSC: (%f/%f/%f)' % unet_f_dsc)
        #
        # cnn_r_dsc = np.mean(ensemble_roi_results)
        # print('Final ROI results DSC: %f' % cnn_r_dsc)
        # cnn_f_dsc = tuple(map(
        #     lambda k: np.mean(
        #         [dsc[k] for dsc in ensemble_seg_results if len(dsc) > k]
        #     ),
        #     range(3)
        # ))
        # print('Final CNN results DSC: (%f/%f/%f)' % cnn_f_dsc)

        ''' <Survival task> '''
        tst_simage_names, tst_survival, tst_features, tst_slices = get_survival_data(options, recession=['GTR'])
        max_survival = np.max(tst_survival)
        n_folds = len(tst_simage_names)
        print('%s[%s] %sStarting leave-one-out (survival)%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
        with open(os.path.join(options['loo_dir'], 'survival_results.csv'), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for i in range(n_folds):
                ''' Training '''
                ini_p = len(tst_simage_names) * i / n_folds
                end_p = len(tst_simage_names) * (i + 1) / n_folds
                # Validation data
                p = tst_simage_names[ini_p:end_p]
                p_features = tst_features[ini_p:end_p]
                p_slices = tst_slices[ini_p:end_p]
                p_survival = tst_survival[ini_p:end_p]
                # Training data
                train_images = np.concatenate([
                    tst_simage_names[:ini_p, :],
                    tst_simage_names[end_p:, :],
                    # simage_names
                ], axis=0)
                train_survival = np.asarray(
                    tst_survival.tolist()[:ini_p] + tst_survival.tolist()[end_p:] # + survival.tolist()
                )
                train_features = np.asarray(
                    tst_features.tolist()[:ini_p] + tst_features.tolist()[end_p:] # + features.tolist()
                )
                train_slices = tst_slices[:ini_p] + tst_slices[end_p:] # + slices

                # Patient info
                p_name = map(lambda pi: pi[0].rsplit('/')[-2], p)

                # Data stuff
                print('%s[%s] %sFold %s(%s%d%s%s/%d)%s' % (
                    c['c'], strftime("%H:%M:%S"),
                    c['g'], c['c'], c['b'], i + 1, c['nc'], c['c'], n_folds, c['nc']
                ))

                print(
                    '%s[%s] %sStarting training (%ssurvival%s)%s' % (
                        c['c'], strftime("%H:%M:%S"),
                        c['g'], c['b'], c['nc'] + c['g'], c['nc']
                    )
                )

                snet = train_survival_function(
                    train_images,
                    train_survival / max_survival,
                    train_features,
                    train_slices,
                    save_path=options['loo_dir'],
                    sufix='-fold%d' % i
                )

                ''' Testing '''
                survival_out = test_survival(
                    snet,
                    p,
                    p_features,
                    p_slices,
                    options['n_slices']
                ) * max_survival
                print(
                    '%s[%s] %sPatient %s%s%s predicted survival = %s%f (%f)%s' % (
                        c['c'], strftime("%H:%M:%S"),
                        c['g'], c['b'], p_name[0], c['nc'],
                        c['g'], survival_out, p_survival, c['nc']
                    )
                )
                csvwriter.writerow([p_name[0], '%f' % float(survival_out)])

    else:
        ''' <Segmentation task> '''

        # TODO: Revert changes
        # ''' Training'''
        # print(
        #     '%s[%s] %sStarting training (%ssegmentation%s)%s' % (
        #         c['c'], strftime("%H:%M:%S"),
        #         c['g'], c['b'], c['nc'] + c['g'], c['nc']
        #     )
        # )
        #
        # net, ensemble = train_seg_function(image_names, label_names, brain_centers, save_path=test_dir)
        #
        # ''' Testing '''
        # print('%s[%s] %sStarting testing (segmentation)%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
        # test_image_names, _ = get_names_from_path(path=test_dir)
        # for i in range(len(test_image_names)):
        #     # Patient stuff
        #     p = test_image_names[i]
        #     p_name = p[0].rsplit('/')[-2]
        #     print('%s[%s] %sPatient %s%s%s %s(%s%d%s%s/%d)%s' % (
        #         c['c'], strftime("%H:%M:%S"),
        #         c['g'], c['b'], p_name, c['nc'],
        #         c['c'], c['b'], i+1, c['nc'], c['c'], len(test_image_names), c['nc']
        #     ))
        #
        #     # > Testing for the tumor ROI
        #     #
        #     # We first test with the ROI segmentation net.
        #     image_unet = test_seg(net, p, p_name + '.unet.test' + sufix, options['nlabels'])
        #
        #     # > Testing for the tumor inside the ROI
        #     #
        #     # All we need to do now is test with the ensemble.
        #     test_seg(
        #         ensemble,
        #         p,
        #         p_name,
        #         options['nlabels'],
        #         mask=image_unet.get_data().astype(np.bool),
        #         verbose=False
        #     )

        ''' <Survival task> '''

        ''' Training'''
        print(
            '%s[%s] %sStarting training (%ssurvival%s)%s' % (
                c['c'], strftime("%H:%M:%S"),
                c['g'], c['b'], c['nc'] + c['g'], c['nc']
            )
        )
        simage_names, survival, features, slices = get_survival_data(options)
        max_survival = np.max(survival)
        snet = train_survival_function(image_names, survival / max_survival, features, slices, save_path=test_dir)

        ''' Testing '''
        with open(os.path.join(test_dir, 'survival_results.csv'), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            names, simage_names, features, slices = get_survival_data(options, test=True)
            survival_out = test_survival(
                snet,
                simage_names,
                features,
                slices,
                options['n_slices']
            ) * max_survival
            for i, (p_name_i, pred_survival_i) in enumerate(zip(names, survival_out)):
                print(
                    '%s[%s] %sPatient %s%s%s %s(%s%d%s%s/%d)%s predicted survival = %s%f%s' % (
                        c['c'], strftime("%H:%M:%S"),
                        c['g'], c['b'], p_name_i, c['nc'],
                        c['c'], c['b'], i + 1, c['nc'], c['c'], len(names), c['nc'],
                        c['g'], pred_survival_i, c['nc']
                    )
                )

                csvwriter.writerow([p_name_i, '%f' % pred_survival_i])


if __name__ == '__main__':
    main()
