from __future__ import print_function
import numpy as np
from nibabel import load as load_nii
from data_manipulation.generate_features import get_patches
from itertools import chain
import keras
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
            lambda (l, lc): np.minimum(get_patches(l, lc, output_size), nlabels - 1, dtype=np.int8),
            zip(labels_generator(label_names), centers)
        )
    y = map(
        lambda y_i: keras.utils.to_categorical(y_i, num_classes=nlabels).reshape((len(y_i), -1, nlabels)),
        y
    )

    return x, y


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
        lambda y_i: keras.utils.to_categorical(y_i, num_classes=nlabels).reshape((len(y_i), -1, nlabels)),
        y
    )

    return [x_d, x_c], y
