from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from nibabel import load as load_nii
from utils import color_codes
from data_creation import get_bounding_blocks, get_blocks, load_images
from data_manipulation.metrics import dsc_seg
from nets import get_brats_unet, get_brats_invunet
from utils import leave_one_out


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
        dest='loo_dir', default='/home/mariano/DATA/Brats17TrainingData',
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
        dest='down_sampling', type=int, default=1,
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
        'invunet': get_brats_invunet
    }

    options = vars(parser.parse_args())
    options['net'] = networks[options['netname']]

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
    return [dsc_seg(gt == l, image == l) for l in labels[1:]]


def train_net(net, image_names, label_names, train_centers, p, sufix, nlabels):
    options = parse_inputs()
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    c = color_codes()
    # Data stuff
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    # Prepare the net hyperparameters
    epochs = options['epochs']
    batch_size = options['batch_size']

    print('%s[%s] %sTraining the network %s(%s%d %sparameters)' %
          (c['c'], strftime("%H:%M:%S"), c['g'], c['nc'], c['b'], net.count_params(), c['nc']))
    # net.summary()
    net_name = os.path.join(patient_path, 'brats2017%s.mdl' % sufix)
    checkpoint = 'brats2017%s.hdf5' % sufix

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
        x, y = get_blocks(
            image_names=image_names,
            label_names=label_names,
            centers=train_centers,
            patch_size=patch_size,
            output_size=patch_size,
            nlabels=nlabels,
            verbose=True
        )
        print('%s- Concatenating the data' % ' '.join([''] * 12))
        x = np.concatenate(x)
        y = np.concatenate(y)
        print('%s-- Using %d blocks of data' % (
            ' '.join([''] * 12),
            len(x)
        ))

        idx = np.random.permutation(range(len(x)))

        x = x[idx]
        y = y[idx]

        print('%s-- X shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, x.shape))))
        print('%s-- Y shape: (%s)' % (' '.join([''] * 12), ', '.join(map(str, y.shape))))

        print('%s%sStarting the training process%s' % (' '.join([''] * 12), c['g'], c['nc']))
        net.fit(x, y, batch_size=batch_size, validation_split=0.25, epochs=epochs, callbacks=callbacks)
        net.load_weights(os.path.join(patient_path, checkpoint))


def test_net(net, p, outputname, nlabels):

    c = color_codes()
    options = parse_inputs()
    p_name = p[0].rsplit('/')[-2]
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    outputname_path = os.path.join(patient_path, outputname + '.nii.gz')
    roiname = os.path.join(patient_path, outputname + '.roi.nii.gz')
    try:
        image = load_nii(outputname_path).get_data()
        load_nii(roiname)
    except IOError:
        # Image loading
        x = np.expand_dims(np.stack(load_images(p), axis=0), axis=0)

        # Network parameters
        conv_blocks = options['conv_blocks']
        n_filters = options['n_filters']
        filters_list = n_filters if len(n_filters) > 1 else n_filters * conv_blocks
        conv_width = options['conv_width']
        kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks

        image_net = options['net'](x.shape[1:], filters_list, kernel_size_list, nlabels)
        # We should copy the weights here
        for l_new, l_orig in zip(image_net.layers[1:], net.layers[1:]):
            l_new.set_weights(l_orig.get_weights())

        # Now we can test
        print('%s[%s] %sTesting the network%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
        # Load only the patient images
        print('%s<Creating the probability map %s%s%s%s - %s%s%s' %
              (c['g'], c['b'], p_name, c['nc'], c['g'], c['b'], outputname, c['nc']))
        pr_maps = image_net.predict(x, batch_size=options['batch_size'])
        image = np.argmax(pr_maps, axis=-1).reshape(x.shape[2:])
        roi_nii = load_nii(p[0])
        roi_nii.get_data()[:] = image
        roi_nii.to_filename(outputname_path)
    return image


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

    dsc_results = list()
    image_names, label_names = get_names_from_path(options)
    print('%s[%s] %s<BRATS 2017 pipeline testing>%s' % (c['c'], strftime("%H:%M:%S"), c['y'], c['nc']))
    print('%s[%s] %sCenter computation%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
    # Block center computation
    centers = map(
        lambda names: get_bounding_blocks(load_nii(names[0]).get_data(), patch_width),
        image_names
    )

    print('%s[%s] %sStarting leave-one-out%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
    for train_centers, i in leave_one_out(centers):
        p = image_names[i]
        # Patient stuff
        p_name = p[0].rsplit('/')[-2]
        patient_path = '/'.join(p[0].rsplit('/')[:-1])

        # Data stuff
        print('%s%sPatient %s%s%s %s(%s%d%s%s/%d)%s' % (
            ' '.join([''] * 12), c['g'],
            c['b'], p_name, c['nc'],
            c['c'], c['b'], i+1, c['nc'], c['c'], len(centers), c['nc']
        ))

        # Training
        input_shape = (image_names.shape[-1],) + patch_size
        net = options['net'](
            input_shape=input_shape,
            filters_list=filters_list,
            kernel_size_list=kernel_size_list,
            nlabels=options['nlabels']
        )
        train_net(
            image_names=image_names,
            label_names=label_names,
            train_centers=train_centers,
            net=net,
            p=p,
            sufix=sufix,
            nlabels=options['nlabels']
        )

        image_cnn_name = os.path.join(patient_path, p_name + '.cnn.test' + sufix)
        try:
            image_cnn = load_nii(image_cnn_name + '.nii.gz').get_data()
        except IOError:
            image_cnn = test_net(net, p, image_cnn_name, options['nlabels'])

        results = check_dsc(label_names[i], image_cnn, options['nlabels'])
        dsc_string = c['g'] + '/'.join(['%f'] * len(results)) + c['nc']
        print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' FCNN DSC: ' +
              dsc_string % tuple(results))

        dsc_results.append(results)

    f_dsc = tuple(map(lambda k: np.array([dsc[k] for dsc in dsc_results if len(dsc) > i]).mean(), range(3)))
    print('Final results DSC: (%f/%f/%f)' % f_dsc)


if __name__ == '__main__':
    main()
