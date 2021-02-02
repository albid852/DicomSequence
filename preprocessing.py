import numpy as np
import pydicom
import cv2
from scipy import interpolate
from typing import Tuple, List
# from DCMSequence import _get_new_ds

# CONVERT INT TO UINT
def convert_int_to_uint(img: np.ndarray) -> np.ndarray:
    """
    Conversion of int16 to uint16
    :param img: numpy array to convert
    :return: numpy array as type uint16
    """
    if img.dtype == np.int16:
        img_min = np.min(img)
        img += abs(img_min)
        return img.astype(np.uint16)


# PREPROCESSING IMAGES
def apply_clahe(img: np.ndarray, clip_lim: int = 40, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Applies CV2's clahe algorithm to an image array.
    :param img: Image to apply clahe to
    :param clip_lim: All bins in the color histogram above the limit are clipped
    and distributed to the existing bins.
    :param tile_grid_size: tile shape
    :return: Clahe image as a numpy array. Still retains whatever dtype the img started with
    """
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=tile_grid_size)
    img = convert_int_to_uint(img)
    clahe_img = clahe.apply(img)
    return clahe_img


def apply_fiji_normalization(img: np.ndarray) -> np.ndarray:
    """
    Applies a fiji normalization to reduce the given image to a 255 range. Looks exactly
    the same as the original image.
    :param img: Image to normalize
    :return: Normalized image as an 8-bit numpy array.
    """
    img_min, img_max = int(np.min(img)), int(np.max(img))
    scale = 256. / (img_max - img_min + 1)
    x = img & 0xffff
    x -= img_min
    x[x < 0] = 0
    x = x * scale + 0.5
    x[x > 255] = 0
    return x.astype(np.uint8)


def apply_cr_normalization(img: np.ndarray) -> np.ndarray:
    """
    Applies the following normalization to reduce the image to a 0-1 range:
    img / (abs(image mean) + 3 * (image standard deviation))
    Then multiplies by 255 and clips the image between 0-255.
    :param img:
    :return: Normalized image as an 8-bit numpy array.
    """
    mu = np.mean(img)
    sigma = np.std(img)
    tmp = img / (abs(mu) + 3 * sigma)
    tmp *= 255
    uint8_img = np.clip(tmp, 0, 255).astype(np.uint8)
    return uint8_img


# RESIZE PNG
def resize(img_list: List[np.ndarray], dsize: Tuple[int, int]) -> None:
    for i in range(len(img_list)):
        img_list[i] = cv2.resize(img_list[i], dsize)


# GET PNG LIST FROM DICOMS
def get_png(dcm_list: List[pydicom.Dataset], clahe: bool = False, norm_alg: int = 1) -> List[np.ndarray]:
    """
    Get list of png images and list of file names by converting the current dicoms in the
    collection to 8-bit using the preferred norm-alg.
    :param dcm_list: list of dicoms to turn to png
    :param clahe: whether or not to perform clahe on the images
    :param norm_alg: which normalization algorithm to use to get the image between 0-255.
    If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
    normalization. norm_alg = 2 is for CR normalization.
    :return: List of file names (not the path), list of png images
    """
    if norm_alg < 0 or norm_alg > 2:
        raise ValueError("norm_alg must be between 0 and 2")

    images = []

    for ds in dcm_list:

        if ds.pixel_array.dtype == np.uint8:
            images.append(ds.pixel_array)

        elif ds.pixel_array.dtype == np.uint16 \
                or ds.pixel_array.dtype == np.int16:
            image = convert_int_to_uint(ds.pixel_array)
            if clahe:
                clip_lim = 40
                tile_grid_size = (8, 8)
                image = apply_clahe(image, clip_lim, tile_grid_size)

            if norm_alg == 0:
                image = np.uint8((image / np.max(image)) * 255)
            elif norm_alg == 1:
                image = apply_fiji_normalization(image)
            elif norm_alg == 2:
                image = apply_cr_normalization(image)

            images.append(image)

    return images


# GET INTERPOLATED IMAGES FROM NUMPY ARRAYS
def interpolate_volume(volume: np.ndarray, num_slices: int = 4) -> np.ndarray:
    """
    Create an interpolated volume from the image stack. This will interpolate slices of
    images between every consecutive pair of slices. The num_slices determines how
    many interpolated slices are between the original slices and the separation between them.
    :param volume: array of images to interpolate between
    :param num_slices: Number of interpolated slices between the original slices
    :return: the entire interpolated volume
    """
    if len(volume.shape) != 3:
        raise ValueError("volume must be of shape (depth, height, width)")
    depth, img_width, img_height = volume.shape
    # set up interpolator
    points = (np.arange(depth), np.arange(img_height), np.arange(img_width))  # (z, y, x)
    rgi = interpolate.RegularGridInterpolator(points, volume)

    # get slices with separation of 1/(num_slices + 1)
    g = np.mgrid[1:num_slices + 1, :img_height, :img_width]
    coords = np.vstack(map(np.ravel, g)).transpose().astype(np.float16)
    coords[:, 0] *= 1 / (num_slices + 1)

    stack = np.zeros((depth + num_slices * (depth - 1), img_height, img_width), dtype=np.uint8)

    # visualize whole volume as slices
    for n in range(depth):
        stack[n * (num_slices + 1)] = volume[n]
        print("SLICE NUMBER:", n + 1)
        if n < depth - 1:
            interp_slices = rgi(coords).reshape((num_slices, img_height, img_width)).astype(np.uint8)
            for i in range(num_slices):
                print("\t", i + 1)
                stack[n * (num_slices + 1) + i + 1] = interp_slices[i]
            coords[:, 0] += 1
    return stack


# CONVERT INT16 TO 8BIT
# def convert_to_8bit(dcm_list: List[pydicom.Dataset], clahe: bool = False, norm_alg: int = 1) -> None:
#     """
#     Convert 16-bit dicoms to 8-bit dicoms.
#     :dcm_list: list of dicoms to convert
#     :clahe: whether or not to use the CLAHE algorithm on the image beforehand
#     :norm_alg: which normalization algorithm to use to get the image between 0-255.
#     If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
#     normalization. norm_alg = 2 is for CR normalization.
#     :return: None
#     """
#     for i, ds in enumerate(dcm_list):
#         if ds.pixel_array.dtype == np.uint16 or ds.pixel_array.dtype == np.int16:
#             new_ds = _get_new_ds(ds, str(i))
#             image = ds.pixel_array
#             image = convert_int_to_uint(image)
#             if clahe:
#                 clip_lim = 40
#                 tile_grid_size = (8, 8)
#                 image = apply_clahe(image, clip_lim, tile_grid_size)
#
#             if norm_alg == 0:
#                 image = np.uint8((image / np.max(image)) * 255)
#             elif norm_alg == 1:
#                 image = apply_fiji_normalization(image)
#             elif norm_alg == 2:
#                 image = apply_cr_normalization(image)
#
#             new_ds.PixelData = image.tobytes()
#
#             dcm_list[i] = new_ds





