from __future__ import print_function
import numpy as np
from nibabel import load as load_nii
from scipy.ndimage.morphology import binary_dilation as imdilate
from itertools import chain, product, izip
from keras.utils import to_categorical
from data_manipulation.generate_features import get_patches, get_mask_voxels


"""
 Challenges 2018 functions

"""


def labels_generator(image_names):
    for patient in image_names:
        yield np.squeeze(np.asarray(load_nii(patient).dataobj))


def load_images(image_names):
    return map(lambda image: norm(np.asarray(load_nii(image).dataobj)), image_names)


def get_bounding_centers(image_names, patch_width, overlap=0, offset=0):
    list_of_centers = map(
        lambda names: get_bounding_blocks(
            load_nii(names[0]).get_data(),
            patch_width,
            overlap=overlap,
            offset=offset
        ),
        image_names
    )
    return list_of_centers


def get_bounding_blocks(mask, block_size, overlap=0, offset=0):
    # Init
    half_size = block_size / 2
    overlap = min(overlap, block_size-1)

    # Bounding box
    min_coord = np.stack(np.nonzero(mask.astype(dtype=np.bool))).min(axis=1) - offset
    max_coord = np.stack(np.nonzero(mask.astype(dtype=np.bool))).max(axis=1) + offset

    # Centers
    global_centers = map(
        lambda (init, end): range(init + half_size, end, block_size - overlap),
        zip(min_coord, max_coord)
    )

    return list(product(*global_centers))


def get_mask_centers(masks, dilation=2):
    list_of_centers = map(lambda mask: get_mask_blocks(mask, dilation=dilation), masks)
    return list_of_centers


def get_mask_blocks(mask, dilation=2):
    return get_mask_voxels(imdilate(mask, iterations=dilation))


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
                lambda image: get_patches(norm(np.asarray(load_nii(image).dataobj)), centers, size),
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
        list_of_centers,
        patch_size,
        datatype=np.float32,
        verbose=False
):
    if verbose:
        print('%s- Loading x' % ' '.join([''] * 12))
    x = filter(lambda z: z.any(), get_patches_list(image_names, list_of_centers, patch_size))
    x = map(lambda x_i: x_i.astype(dtype=datatype), x)
    return x


def get_labels(label_names, list_of_centers, nlabels, verbose=False):
    if verbose:
        print('%s- Loading y' % ' '.join([''] * 12))
    y = map(
            lambda (labels, centers): np.minimum(
                map(lambda c: labels[c], centers),
                nlabels - 1,
                dtype=np.int8
            ),
            izip(labels_generator(label_names), list_of_centers)
        )
    y = map(
        lambda y_i: to_categorical(y_i, num_classes=nlabels),
        y
    )
    return y


def get_patch_labels(label_names, list_of_centers, output_size, nlabels, verbose=False):
    if verbose:
        print('%s- Loading y' % ' '.join([''] * 12))
    y = map(
            lambda (labels, centers): np.minimum(
                get_patches(labels, centers, output_size),
                nlabels - 1,
                dtype=np.int8
            ),
            izip(labels_generator(label_names), list_of_centers)
        )
    y = map(
        lambda y_i: to_categorical(y_i, num_classes=nlabels).reshape((len(y_i), -1, nlabels)),
        y
    )
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


def downsample_centers(list_of_centers, step):
    n_images = len(list_of_centers)
    map(
        a,
        list_of_centers
    )

    return final_centers


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
    return (voting > 0.5).astype(dtype=datatype), voting
