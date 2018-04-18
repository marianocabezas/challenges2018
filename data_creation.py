from __future__ import print_function
import sys
from operator import itemgetter
import numpy as np
from nibabel import load as load_nii
from data_manipulation.generate_features import get_mask_voxels, get_patches, get_rolling_patches
from itertools import izip, chain
from scipy.ndimage.morphology import binary_dilation as imdilate
from scipy.stats import entropy
from numpy import logical_and as log_and
from numpy import logical_or as log_or
from numpy import logical_not as log_not
import keras
from keras.utils import to_categorical
from itertools import product


def clip_to_roi(images, roi):
    # We clip with padding for patch extraction
    min_coord = np.stack(np.nonzero(roi.astype(dtype=np.bool))).min(axis=1)
    max_coord = np.stack(np.nonzero(roi.astype(dtype=np.bool))).max(axis=1)

    clip = np.array([(min_c, max_c) for min_c, max_c in zip(min_coord, max_coord)], dtype=np.uint8)
    im_clipped = images[:, clip[0, 0]:clip[0, 1], clip[1, 0]:clip[1, 1], clip[2, 0]:clip[2, 1]]

    return im_clipped, clip


def labels_generator(image_names):
    for patient in image_names:
        yield np.squeeze(load_nii(patient).get_data())


def load_norm_list(image_list):
    return [norm(load_nii(image).get_data()) for image in image_list]


def load_patch_batch_generator_test(
        image_names,
        centers,
        batch_size,
        size,
        preload=False,
        datatype=np.float32,
):
    while True:
        n_centers = len(centers)
        image_list = load_norm_list(image_names) if preload else image_names
        for i in range(0, n_centers, batch_size):
            print('%f%% tested (step %d)' % (100.0*i/n_centers, (i/batch_size)+1), end='\r')
            sys.stdout.flush()
            x = get_patches_list([image_list], [centers[i:i + batch_size]], size, preload)
            x = np.concatenate(x).astype(dtype=datatype)
            yield x


def load_masks(mask_names):
    for image_name in mask_names:
        yield np.squeeze(load_nii(image_name).get_data().astype(dtype=np.bool))


def get_cnn_centers(names, labels_names, balanced=True, neigh_width=15):
    rois = list(load_masks(names))
    rois_p = list(load_masks(labels_names))
    rois_p_neigh = [log_and(log_and(imdilate(roi_p, iterations=neigh_width), log_not(roi_p)), roi)
                    for roi, roi_p in izip(rois, rois_p)]
    rois_n_global = [log_and(roi, log_not(log_or(roi_pn, roi_p)))
                     for roi, roi_pn, roi_p in izip(rois, rois_p_neigh, rois_p)]
    rois = list()
    for roi_pn, roi_ng, roi_p in izip(rois_p_neigh, rois_n_global, rois_p):
        # The goal of this for is to randomly select the same number of nonlesion and lesion samples for each image.
        # We also want to make sure that we select the same number of boundary negatives and general negatives to
        # try to account for the variability in the brain.
        n_positive = np.count_nonzero(roi_p)
        if balanced:
            roi_pn[roi_pn] = np.random.permutation(xrange(np.count_nonzero(roi_pn))) < n_positive / 2
        roi_ng[roi_ng] = np.random.permutation(xrange(np.count_nonzero(roi_ng))) < n_positive / 2
        rois.append(log_or(log_or(roi_ng, roi_pn), roi_p))

    # In order to be able to permute the centers to randomly select them, or just shuffle them for training, we need
    # to keep the image reference with the center. That's why we are doing the next following lines of code.
    centers_list = [get_mask_voxels(roi) for roi in rois]
    idx_lesion_centers = np.concatenate([np.array([(i, c) for c in centers], dtype=object)
                                         for i, centers in enumerate(centers_list)])

    return idx_lesion_centers


"""
 Challenges 2018 functions

"""


def load_images(image_names):
    return map(lambda image: norm(load_nii(image).get_data()), image_names)


def get_bounding_blocks(mask, block_size, overlap=0):
    # Init
    half_size = block_size / 2
    overlap = min(overlap, block_size-1)

    # Bounding box
    min_coord = np.stack(np.nonzero(mask.astype(dtype=np.bool))).min(axis=1)
    max_coord = np.stack(np.nonzero(mask.astype(dtype=np.bool))).max(axis=1)

    # Centers
    global_centers = map(
        lambda (init, end): range(init + half_size, end, block_size - overlap),
        zip(min_coord, max_coord)
    )

    return list(product(*global_centers))


def centers_and_idx(centers, n_images):
    # This function is used to decompress the centers with image references into image indices and centers.
    centers = [list(map(lambda z: tuple(z[1]) if z[0] == im else [], centers)) for im in range(n_images)]
    idx = [map(lambda (a, b): [a] if b else [], enumerate(c)) for c in centers]
    centers = [filter(bool, c) for c in centers]
    idx = list(chain(*[chain(*i) for i in idx]))
    return centers, idx


def get_patches_list(list_of_image_names, centers_list, size):
    patch_list = [
        np.stack(
            map(
                lambda image: get_patches(norm(load_nii(image).get_data()), centers, size),
                image_names
            ),
            axis=1
        )
        for image_names, centers in zip(list_of_image_names, centers_list) if centers
    ]
    return patch_list


def norm(image):
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


def get_blocks(
        image_names,
        label_names,
        centers,
        patch_size,
        output_size,
        nlabels,
        datatype=np.float32,
        verbose=False
):
    if verbose:
        print('%s- Loading x' % ' '.join([''] * 12))
    x = filter(lambda z: z.any(), get_patches_list(image_names, centers, patch_size))
    x = map(lambda x_i: x_i.astype(dtype=datatype), x)
    if verbose:
        print('%s- Loading y' % ' '.join([''] * 12))
    y = map(
            lambda (l, lc): np.asarray(get_patches(l, lc, output_size), dtype=np.int8),
            zip(labels_generator(label_names), centers)
        )
    y = map(
        lambda y_i: keras.utils.to_categorical(y_i, num_classes=nlabels).reshape((len(y_i), -1, nlabels)),
        y
    )

    return x, y
