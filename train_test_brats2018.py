from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from nibabel import load as load_nii
from utils import color_codes
from data_creation import get_mask_centers, get_bounding_centers, get_mask_blocks
from data_creation import get_patch_labels, get_data, get_labels, load_images
from data_manipulation.metrics import dsc_seg
from nets import get_brats_unet, get_brats_invunet, get_brats_cnn


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-t', '--train',
        dest='train_dir', default=None,
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
        help='Number of maximum epochs for training'
    )
    parser.add_argument(
        '-v', '--validation-rate',
        action='store', dest='val_rate', type=float, default=0.25,
        help='Rate of training samples used for validation'
    )
    parser.add_argument(
        '-u', '--unbalanced',
        action='store_false', dest='balanced', default=True,
        help='Data balacing for training'
    )
    parser.add_argument(
        '-p', '--preload',
        action='store_true', dest='preload', default=False,
        help='Whether samples are preloaded on RAM or not'
    )
    parser.add_argument(
        '-P', '--patience',
        dest='patience', type=int, default=5,
        help='Maximum number of epochs without validation accuracy improvement'
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

    networks = {
        'unet': get_brats_unet,
        'invunet': get_brats_invunet,
    }

    options = vars(parser.parse_args())
    options['net'] = networks[options['netname']]

    if options['netname'] is 'roinet':
        options['nlabels'] = 2

    return options


def get_names_from_path(options):
    path = options['dir_train'] if options['train_dir'] is not None else options['loo_dir']

    directories = filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])
    patients = sorted(directories)

    # Prepare the names
    def get_names(sufix):
        return map(lambda p: os.path.join(path, p, p.split('/')[-1] + sufix), patients)
    flair_names = get_names(options['flair']) if options['use_flair'] else None
    # flair_names = [os.path.join(path, p, p.split('/')[-1] + options['flair']) for p in patients]
    t2_names = get_names(options['t2']) if options['use_t2'] else None
    # t2_names = [os.path.join(path, p, p.split('/')[-1] + options['t2']) for p in patients]
    t1_names = get_names(options['t1']) if options['use_t1'] else None
    # t1_names = [os.path.join(path, p, p.split('/')[-1] + options['t1']) for p in patients]
    t1ce_names = get_names(options['t1ce']) if options['use_t1ce'] else None
    # t1ce_names = [os.path.join(path, p, p.split('/')[-1] + options['t1ce']) for p in patients]

    # label_names = np.array([os.path.join(path, p, p.split('/')[-1] + options['labels']) for p in patients])
    label_names = np.array(get_names(options['labels']))
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names


def check_dsc(gt_name, image, nlabels):
    gt_nii = load_nii(gt_name)
    gt = np.minimum(gt_nii.get_data(), nlabels-1).astype(dtype=np.uint8)
    labels = np.unique(gt.flatten())
    gt_nii.uncache()
    return [dsc_seg(gt == l, image == l) for l in labels[1:]]


def train_net(net, image_names, label_names, train_centers, p, sufix, nlabels, seg=False):
    options = parse_inputs()
    conv_blocks = options['conv_blocks']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    c = color_codes()
    # Data stuff
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    # Prepare the net hyperparameters
    epochs = options['epochs']
    batch_size = options['batch_size']

    net_name = os.path.join(patient_path, 'brats2018%s.mdl' % sufix)
    checkpoint = 'brats2018%s.hdf5' % sufix

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=options['patience']
        ),
        ModelCheckpoint(
            os.path.join(patient_path, checkpoint),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    net.save(net_name)

    try:
        net.load_weights(os.path.join(patient_path, checkpoint))
    except IOError:
        print(
            '%s[%s] %sTraining the network %s%s %s(%s%d %sparameters)' % (
                c['c'], strftime("%H:%M:%S"), c['g'], c['b'], 'Unet' if not seg else 'CNN', c['nc'],
                c['b'], net.count_params(), c['nc']
            )
        )
        # net.summary()
        if not seg:
            x = get_data(
                image_names=image_names,
                list_of_centers=train_centers,
                patch_size=patch_size,
                verbose=True,
            )
            print('%s- Concatenating the data' % ' '.join([''] * 12))
            x = np.concatenate(x)
        else:
            x_d = get_data(
                image_names=image_names,
                list_of_centers=train_centers,
                patch_size=(options['conv_blocks_seg'] * 2 + 3,) * 3,
                verbose=True,
            )
            print('%s- Concatenating the data' % ' '.join([''] * 12))
            x_d = np.concatenate(x_d)
            print('%s- %d samples' % (' '.join([''] * 12), len(x_d)))
            x_c = get_data(
                image_names=image_names,
                list_of_centers=train_centers,
                patch_size=(3, 3, 3),
                verbose=True,
            )
            print('%s- Concatenating the data' % ' '.join([''] * 12))
            x_c = np.concatenate(x_c)
            print('%s- %d samples' % (' '.join([''] * 12), len(x_c)))
            x = [x_c, x_d]

        if not seg:
            y = get_patch_labels(
                label_names=label_names,
                list_of_centers=train_centers,
                output_size=patch_size,
                nlabels=nlabels,
                verbose=True
            )
        else:
            y = get_labels(
                label_names=label_names,
                list_of_centers=train_centers,
                nlabels=nlabels,
                verbose=True
            )
        print('%s- Concatenating the labels' % ' '.join([''] * 12))
        y = np.concatenate(y)
        print('%s-- Using %d blocks of data' % (
            ' '.join([''] * 12),
            len(y)
        ))
        if type(x) is not list:
            print('%s-- X shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, x.shape))))
        y_message = '%s-- Y shape: (%s)'
        print(y_message % (' '.join([''] * 12), ', '.join(map(str, y.shape))))

        print('%s- Randomising the training data' % ' '.join([''] * 12))
        idx = np.random.permutation(range(len(y)))

        x = x[idx] if type(x) is not list else map(lambda xi: xi[idx], x)
        y = y[idx]

        print('%s%sStarting the training process%s' % (' '.join([''] * 12), c['g'], c['nc']))
        net.fit(x, y, batch_size=batch_size, validation_split=0.25, epochs=epochs, callbacks=callbacks)
        net.load_weights(os.path.join(patient_path, checkpoint))


def test_net(net, p, outputname, nlabels, mask=None, verbose=True):

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
        x = np.expand_dims(np.stack(load_images(p), axis=0), axis=0)
        if mask is None:
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
                print('%s%s<Creating the probability map %s%s%s%s - %s%s%s %s>%s' % (
                    ''.join([' '] * 14),
                    c['g'],
                    c['b'], p_name, c['nc'], c['g'],
                    c['b'], outputname, c['nc'],
                    c['g'], c['nc']
                ))
            pr_maps = image_net.predict(x, batch_size=options['test_size'])
            image = np.argmax(pr_maps, axis=-1).reshape(x.shape[2:])
        else:
            image = np.zeros_like(mask, dtype=np.int8)
            options = parse_inputs()
            conv_blocks = options['conv_blocks_seg']
            test_centers = get_mask_blocks(mask)
            x_d = get_data(
                image_names=[p],
                list_of_centers=[test_centers],
                patch_size=(conv_blocks * 2 + 3,) * 3,
                verbose=True,
            )
            if verbose:
                print('%s- Concatenating the data x_d' % ' '.join([''] * 12))
            x_d = np.concatenate(x_d)
            x_c = get_data(
                image_names=[p],
                list_of_centers=[test_centers],
                patch_size=(3, 3, 3),
                verbose=True,
            )
            if verbose:
                print('%s- Concatenating the data x_c' % ' '.join([''] * 12))
            x_c = np.concatenate(x_c)
            x = [x_c, x_d]
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
    patch_size = (patch_width, patch_width, patch_width)
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
    train_data, _ = get_names_from_path(options)

    unet_seg_results = list()
    unet_roi_results = list()
    cnn_seg_results = list()
    cnn_roi_results = list()
    image_names, label_names = get_names_from_path(options)
    print('%s[%s] %s<BRATS 2018 pipeline testing>%s' % (c['c'], strftime("%H:%M:%S"), c['y'], c['nc']))
    print('%s[%s] %sCenter computation%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
    # Block center computation
    overlap = 0 if options['netname'] != 'roinet' else patch_width / 4
    brain_centers = get_bounding_centers(image_names, patch_width, overlap)

    print('%s[%s] %sStarting leave-one-out%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
    for i in range(len(image_names)):
        # Data separation
        p = image_names[i]
        train_images = np.asarray(image_names.tolist()[:i] + image_names.tolist()[i + 1:])
        train_labels = np.asarray(label_names.tolist()[:i] + label_names.tolist()[i + 1:])
        train_centers = brain_centers[:i] + brain_centers[i + 1:]
        # Patient stuff
        p_name = p[0].rsplit('/')[-2]
        patient_path = '/'.join(p[0].rsplit('/')[:-1])

        # Data stuff
        print('%s%sPatient %s%s%s %s(%s%d%s%s/%d)%s' % (
            ' '.join([''] * 12), c['g'],
            c['b'], p_name, c['nc'],
            c['c'], c['b'], i+1, c['nc'], c['c'], len(brain_centers), c['nc']
        ))

        '''Tumor ROI stuff'''
        # Training for the ROI
        input_shape = (image_names.shape[-1],) + patch_size
        net = options['net'](
            input_shape=input_shape,
            filters_list=filters_list,
            kernel_size_list=kernel_size_list,
            nlabels=options['nlabels']
        )
        train_net(
            image_names=train_images,
            label_names=train_labels,
            train_centers=train_centers,
            net=net,
            p=p,
            sufix=sufix,
            nlabels=options['nlabels']
        )

        # Testing for the ROI
        image_unet = test_net(net, p, p_name + '.unet.test' + sufix, options['nlabels'])

        seg_dsc = check_dsc(label_names[i], image_unet.get_data(), options['nlabels'])
        roi_dsc = check_dsc(label_names[i], image_unet.get_data().astype(np.bool), 2)

        dsc_string = c['g'] + '/'.join(['%f'] * len(seg_dsc)) + c['nc'] + ' (%f)'
        print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' Unet DSC: ' +
              dsc_string % tuple(seg_dsc + roi_dsc))

        unet_seg_results.append(seg_dsc)
        unet_roi_results.append(roi_dsc)

        '''Tumor segmentation stuff'''
        # Training for the tumor inside the ROI
        # First we should retest each training image and get the test tumor mask
        # and join it with the GT mask. This new mask after dilation will give us
        # the training centers.
        # print('%s%s<Creating the tumor masks for the training data>%s' % (
        #             ''.join([' '] * 14), c['g'], c['nc']
        #         ))
        # masks = map(
        #     lambda (images, labels): np.logical_or(
        #         test_net(
        #             net,
        #             images,
        #             p_name + '.test' + sufix,
        #             options['nlabels'],
        #             verbose=False
        #         ).get_data().astype(np.bool),
        #         load_nii(labels).get_data().astype(np.bool)
        #     ),
        #     zip(train_images, train_labels)
        # )

        # print('%s- Extracting centers from the tumor ROI' % ' '.join([''] * 12))
        # train_centers = get_mask_centers(masks)
        # train_centers = map(
        #     lambda centers:  map(
        #         tuple,
        #         np.random.permutation(centers)[::options['down_sampling']].tolist()
        #     ),
        #     train_centers
        # )
        #
        # dense_size = options['dense_size']
        # conv_blocks_seg = options['conv_blocks_seg']
        # net = get_brats_cnn(
        #     n_channels=image_names.shape[-1],
        #     filters_list=n_filters * conv_blocks_seg,
        #     kernel_size_list=[conv_width] * conv_blocks_seg,
        #     nlabels=options['nlabels'],
        #     dense_size=dense_size
        # )
        # train_net(
        #     image_names=train_images,
        #     label_names=train_labels,
        #     train_centers=train_centers,
        #     net=net,
        #     p=p,
        #     sufix='%s.d%d' % (sufix, dense_size),
        #     nlabels=options['nlabels'],
        #     seg=True
        # )
        # # Testing for the tumor inside the ROI
        # image_cnn = test_net(
        #     net,
        #     p,
        #     p_name + '.cnn.test' + sufix,
        #     options['nlabels'],
        #     mask=image_unet.get_data().astype(np.bool)
        # )
        #
        # seg_dsc = check_dsc(label_names[i], image_cnn.get_data(), options['nlabels'])
        # roi_dsc = check_dsc(label_names[i], image_cnn.get_data().astype(np.bool), 2)
        #
        # dsc_string = c['g'] + '/'.join(['%f'] * len(seg_dsc)) + c['nc'] + ' (%f)'
        # print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' CNN DSC: ' +
        #       dsc_string % tuple(seg_dsc + roi_dsc))
        #
        # cnn_seg_results.append(seg_dsc)
        # cnn_roi_results.append(roi_dsc)

    unet_r_dsc = np.mean(unet_roi_results)
    print('Final ROI results DSC: %f' % unet_r_dsc)
    unet_f_dsc = tuple(map(
        lambda k: np.mean(
            [dsc[k] for dsc in unet_seg_results if len(dsc) > k]
        ),
        range(3)
    ))
    print('Final Unet results DSC: (%f/%f/%f)' % unet_f_dsc)
    # cnn_r_dsc = np.mean(cnn_roi_results)
    # print('Final ROI results DSC: %f' % cnn_r_dsc)
    # cnn_f_dsc = tuple(map(
    #     lambda k: np.mean(
    #         [dsc[k] for dsc in cnn_seg_results if len(dsc) > k]
    #     ),
    #     range(3)
    # ))
    # print('Final CNN results DSC: (%f/%f/%f)' % cnn_f_dsc)


if __name__ == '__main__':
    main()
