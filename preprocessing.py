import numpy as np
import pydicom
import cv2
from scipy import interpolate
from typing import Tuple, List
import math

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
    else:
        return img


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
def resize(img_list: List[np.ndarray], dsize: Tuple[int, int]) -> List[np.ndarray]:
    return [cv2.resize(img, dsize) for img in img_list]


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
def interpolate_imgs2vol(volume: np.ndarray, num_slices: int = 4) -> np.ndarray:
    """
    Create an interpolated volume from the image stack. This will interpolate slices of
    images between every consecutive pair of slices. The num_slices determines how
    many interpolated slices are between the original slices and the separation between them.
    :param volume: array of images to interpolate between
    :param num_slices: Number of interpolated slices between the original slices
    :return: the entire interpolated volume
    """
    if len(volume.shape) != 3:
        raise ValueError(f"volume must be of shape (depth, height, width), not {volume.shape}")
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


# GET INTERPOLATED IMAGES FROM DICOMS
def interpolate_dcm2vol(dcm_list: List[pydicom.dataset.FileDataset],
                        interp_factor: float,
                        clahe: bool = False, norm_alg: int = 1) -> np.ndarray:
    """
    Create an interpolated volume from the dicom list. This will interpolate slices of
    images between every consecutive pair of slices. Number of slices between is based on
    the DICOM header information, which will either have Spacing Between Slices or Slice
    Location. Either way, we end up with a 1 mm spacing between each slice in the final
    interpolated volume
    :param dcm_list: list of pydicom.dataset.FileDataset
    :param interp_factor: factor of interpolation, divide the separation by this for method 0
    :param clahe: whether or not to perform clahe on the images
    :param norm_alg: which normalization algorithm to use to get the image between 0-255.
    If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
    normalization. norm_alg = 2 is for CR normalization.
    :return: the entire interpolated volume
    """
    if len(dcm_list) == 0:
        return np.array([])

    volume = np.array(get_png(dcm_list, clahe=clahe, norm_alg=norm_alg))
    if len(volume.shape) != 3:
        raise ValueError(f"volume must be of shape (depth, height, width), not {volume.shape}")

    depth, img_width, img_height = volume.shape

    # set up interpolator
    points = (np.arange(depth), np.arange(img_height), np.arange(img_width))  # (z, y, x)
    rgi = interpolate.RegularGridInterpolator(points, volume)

    # check whether to use Spacing between slices, Slice Location or neither
    method = None  # 0 = slice location,  1 = spacing between slices,  2 = not given
    prev = None
    n = 0
    while n < depth:
        d = dcm_list[n]
        if hasattr(d, "SpacingBetweenSlices"):
            if method == 0 or method == 2:
                RuntimeError(f"""DICOM {n} does not utilize Spacing Between Slices. 
                                 All DICOMS must use the same positioning technique""")
            if prev is not None and prev != d.SpacingBetweenSlices:
                RuntimeError(f"""DICOM {n} using Spacing Between Slices does not have the 
                                 same spacing as previous ones.""")
            elif prev is None:
                prev = d.SpacingBetweenSlices
            method = 1
        elif hasattr(d, "SliceLocation"):
            if method == 1 or method == 2:
                RuntimeError(f"""DICOM {n} does not utilize Slice Location. 
                                 All DICOMS must use the same positioning technique""")
            method = 0
        else:
            if method == 0:
                RuntimeError(f"""DICOM {n} utilizes Slice Location, but previous ones did not. 
                                 All DICOMS must use the same positioning technique""")
            if method == 1:
                RuntimeError(f"""DICOM {n} utilizes Spacing Between Slices, but previous ones did not. 
                                 All DICOMS must use the same positioning technique""")

            method = 2
        n += 1

    # solving the problem
    if method == 2:
        # Conduct interpolation with constant 4 slices between
        # get slices with desired separation
        g = np.mgrid[1:5, :img_height, :img_width]  # 4 slices between (5 - 1)
        coords = np.vstack(map(np.ravel, g)).transpose().astype(np.float16)  # [z, y, x] between 2 slices
        coords[:, 0] *= 1 / 5  # scale to be properly between the slices

        # final interpolated volume but empty
        stack = np.zeros((depth + 4 * (depth - 1), img_height, img_width), dtype=np.uint8)

        for i in range(depth):
            stack[i * 5] = volume[i]
            print("SLICE NUMBER:", i + 1)
            if i < depth - 1:
                interp_slices = rgi(coords).reshape((4, img_height, img_width)).astype(np.uint8)
                for j in range(4):
                    stack[i * 5 + j + 1] = interp_slices[j]
                coords[:, 0] += 1
        return stack

    elif method == 1:

        num_slices = math.floor(dcm_list[0].SpacingBetweenSlices)

        # final interpolated volume but empty
        stack = np.zeros((depth + num_slices * (depth - 1), img_height, img_width), dtype=np.uint8)

        # Conduct interpolation with spacing between slices. Is constant for one image sequence
        # get slices with desired separation
        g = np.mgrid[1:(num_slices + 1), :img_height, :img_width]  # 4 slices between (5 - 1)
        coords = np.vstack(map(np.ravel, g)).transpose().astype(np.float16)  # [z, y, x] between 2 slices
        coords[:, 0] *= 1 / (num_slices + 1)  # scale to be properly between the slices

        for i in range(depth):
            stack[i * (num_slices + 1)] = volume[i]
            print("SLICE NUMBER:", i + 1)
            if i < depth - 1:
                interp_slices = rgi(coords).reshape((num_slices, img_height, img_width)).astype(np.uint8)
                for j in range(num_slices):
                    stack[i * (num_slices + 1) + j + 1] = interp_slices[j]
                coords[:, 0] += 1
        return stack

    elif method == 0:
        total_slices_between = 0

        # get total number of slices inbetween
        for i in range(depth):
            d = dcm_list[i]
            if i < depth - 1:
                d2 = dcm_list[i + 1]
                num_slices = math.floor(abs(d.SliceLocation - d2.SliceLocation) * interp_factor)
                total_slices_between += num_slices
        # final interpolated volume but empty
        stack = np.zeros((depth + total_slices_between, img_height, img_width), dtype=np.uint8)

        elapsed = 0
        for i in range(depth):
            print("SLICE NUMBER:", i + 1)
            d = dcm_list[i]
            if i < depth - 1:
                d2 = dcm_list[i + 1]
                num_slices = math.floor(abs(d.SliceLocation - d2.SliceLocation) * interp_factor)  # for 1 mm separation

                stack[elapsed] = volume[i]
                elapsed += 1

                # get coords of interpolated points
                g = np.mgrid[1:(num_slices + 1), :img_height, :img_width]
                coords = np.vstack(map(np.ravel, g)).transpose().astype(np.float16)  # [z, y, x] between 2 slices
                coords[:, 0] *= 1 / (num_slices + 1)  # scale to be properly between the slices
                coords[:, 0] += i

                interp_slices = rgi(coords).reshape((num_slices, img_height, img_width)).astype(np.uint8)
                for j in range(num_slices):
                    stack[elapsed] = interp_slices[j]
                    elapsed += 1
            else:
                # cap off the stack with the last image
                stack[-1] = volume[-1]

        return stack
