#!/usr/bin/python
from __future__ import print_function
import os
from time import strftime
import numpy as np
from nibabel import load as load_nii
from utils import color_codes, get_biggest_region
from data_creation import get_mask_blocks, get_data, load_images
from nets import get_brats_unet, get_brats_ensemble, get_brats_nets


def main():
    # Init
    c = color_codes()
    nlabels = 5

    # Prepare the names
    flair_name = '/data/flair.nii.gz'
    t2_name = '/data/t2.nii.gz'
    t1_name = '/data/t1.nii.gz'
    t1ce_name = '/data/t1ce.nii.gz'

    image_names = [flair_name, t2_name, t1_name, t1ce_name]

    ''' Unet stuff '''
    # Data loading
    x = np.expand_dims(np.stack(load_images(image_names), axis=0), axis=0)

    # Network loading and testing
    print('%s[%s] %sTesting the Unet%s' % (c['c'], strftime("%H:%M:%S"), c['g'], c['nc']))
    net = get_brats_unet(x.shape[1:], [32] * 5, [3] * 5, nlabels)
    net.load_weights('/usr/local/models/brats18-unet.hdf5')
    pr_maps = net.predict(x)
    image = np.argmax(pr_maps, axis=-1).reshape(x.shape[2:])
    mask = get_biggest_region(image).astype(np.bool)

    ''' Ensemble stuff '''
    # Init
    image = np.zeros_like(mask, dtype=np.int8)

    # Data loading
    test_centers = get_mask_blocks(mask)
    x = get_data(
        image_names=[image_names],
        list_of_centers=[test_centers],
        patch_size=(9,) * 3,
        verbose=True,
    )
    print('%s- Concatenating the data x' % ' '.join([''] * 12))
    x = np.concatenate(x)

    # Networks loading
    nets, unet, cnn, fcnn, ucnn = get_brats_nets(
        n_channels=len(image_names),
        filters_list=[32] * 3,
        kernel_size_list=[3] * 3,
        nlabels=nlabels,
        dense_size=256
    )

    ensemble = get_brats_ensemble(
        n_channels=len(image_names),
        n_blocks=3,
        unet=unet,
        cnn=cnn,
        fcnn=fcnn,
        ucnn=ucnn,
        nlabels=nlabels
    )

    nets.load_weights('/usr/local/models/brats18-nets.hdf5')
    ensemble.load_weights('/usr/local/models/brats18-ensemble.hdf5')

    # Network testing and results saving
    pr_maps = ensemble.predict(x)
    [x, y, z] = np.stack(test_centers, axis=1)
    image[x, y, z] = np.argmax(pr_maps, axis=1).astype(dtype=np.int8)

    if not os.path.isdir('/data/results'):
        os.mkdir('/data/results')
    roi_nii = load_nii(flair_name)
    roi_nii.get_data()[:] = image
    roi_nii.to_filename('/data/results/tumor_NVICOROB_class.nii.gz')


if __name__ == '__main__':
    main()