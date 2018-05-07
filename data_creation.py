from __future__ import print_function
import numpy as np
from nibabel import load as load_nii
from data_manipulation.generate_features import get_patches
from itertools import chain
from keras.utils import to_categorical
from itertools import product


"""
 Challenges 2018 functions

"""


def labels_generator(image_names):
    for patient in image_names:
        yield np.squeeze(load_nii(patient).get_data())


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


def get_data(
        image_names,
        centers,
        patch_size,
        datatype=np.float32,
        verbose=False
):
    if verbose:
        print('%s- Loading x' % ' '.join([''] * 12))
    x = filter(lambda z: z.any(), get_patches_list(image_names, centers, patch_size))
    x = map(lambda x_i: x_i.astype(dtype=datatype), x)
    return x


def get_labels(label_names, centers, output_size, nlabels, roinet=False, verbose=False):
    if verbose:
        print('%s- Loading y' % ' '.join([''] * 12))
    y = map(
            lambda (l, lc): np.minimum(get_patches(l, lc, output_size), nlabels - 1, dtype=np.int8),
            zip(labels_generator(label_names), centers)
        )
    if not roinet:
        y = map(
            lambda y_i: to_categorical(y_i, num_classes=nlabels).reshape((len(y_i), -1, nlabels)),
            y
        )
    else:
        y_tumor = map(
            lambda (l, lc): to_categorical(
                np.minimum(map(lambda c: l[c], nlabels - 1, lc), dtype=np.int8), num_classes=nlabels
            ),
            zip(labels_generator(label_names), centers)
        )
        y_block = map(
            lambda y_i: to_categorical(y_i, num_classes=nlabels).reshape((len(y_i), -1, nlabels)),
            y
        )
        y = (y_tumor, y_block)
    return y


def get_patches_roi(
        image_names,
        label_names,
        roi,
        nblocks,
        nlabels,
        datatype=np.float32,
        verbose=False
):
    centers = get_bounding_blocks(roi, 3, 2)
    if verbose:
        print('%s- Loading x' % ' '.join([''] * 12))
    x_d = filter(lambda z: z.any(), get_patches_list(image_names, centers, (3, 3, 3)))
    x_d = map(lambda x_i: x_i.astype(dtype=datatype), x_d)

    x_c = filter(lambda z: z.any(), get_patches_list(image_names, centers, (nblocks * 2 + 3,) * 3))
    x_c = map(lambda x_i: x_i.astype(dtype=datatype), x_c)

    if verbose:
        print('%s- Loading y' % ' '.join([''] * 12))
    y = map(
            lambda (l, lc): np.minimum(map(lambda c: l[c], lc), nlabels - 1, dtype=np.int8),
            zip(labels_generator(label_names), centers)
        )
    y = map(
        lambda y_i: to_categorical(y_i, num_classes=nlabels).reshape((len(y_i), -1, nlabels)),
        y
    )

    return [x_d, x_c], y


def majority_voting_patches(patches, image_size, patch_size, centers, datatype=np.uint8):
    """
    :param patches:
    :param image_size:
    :param patch_size:
    :param centers:
    :param datatype:
    :return:
    """

    ''' Init '''
    # Initial image.
    image = np.zeros(image_size, dtype=np.float32)
    image_explored = np.zeros(image_size, dtype=np.float32)

    # We assume that the centers and their corresponding patches are inside the image boundaries.
    slices = map(
        lambda center: map(
            lambda (c_idx, s_idx): slice(c_idx - s_idx / 2, c_idx + (s_idx - s_idx / 2)),
            zip(center, patch_size)
        ),
        centers
    )
    for slice_i, patch_i in zip(slices, patches):
        image[slice_i] += patch_i[:, -1].reshape(patch_size)
        image_explored[slice_i] += np.ones(patch_size)
    image_explored[image_explored == 0] = 1
    voting = np.divide(image, image_explored)
    return (voting > 0.5).astype(dtype=datatype)
