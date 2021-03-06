import os
import numpy as np
import pydicom
import glob
import cv2
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
from preprocessing import apply_clahe, apply_fiji_normalization, \
    apply_cr_normalization, convert_int_to_uint, interpolate_imgs2vol, \
    interpolate_dcm2vol, get_png
from PlotDCM import plot_comparisons, multi_slice_viewer

def _get_new_ds(ds: pydicom.dataset.Dataset, name: str):
    """
    Create a new dicom header. Creates a new dataset object and sets all existing
    fields in ds.
    :param ds: old dataset to base this new one off of
    :param name: name for this file
    :return: The new dicom dataset object with all header information. No pixel data.
    """
    file_meta = ds.file_meta

    new_ds = pydicom.dataset.FileDataset(name, {}, file_meta=file_meta,
                                         preamble=b"\0" * 128, is_implicit_VR=ds.is_implicit_VR,
                                         is_little_endian=ds.is_little_endian)

    a = [attr for attr in dir(ds) if not attr.startswith('__') and not callable(getattr(ds, attr))]

    for attr in a:
        if attr in ['_character_set', 'is_original_encoding', 'pixel_array',
                    'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
                    'PixelData']:
            continue
        else:
            new_ds.__setattr__(attr, ds.__getattr__(attr))

    # fix bits
    new_ds.BitsAllocated = 8
    new_ds.BitsStored = 8
    new_ds.HighBit = 7
    new_ds.PixelRepresentation = 0

    return new_ds


class DcmSequence:

    def __init__(self):
        self.dcm_files = []
        self.collection = []
        self.mask_files = []
        self.masks = []

    def __str__(self):
        s = "-----------------------------------------\n DICOMS\n"
        for i in range(len(self.dcm_files)):
            img_shape = self.collection[i].pixel_array.shape
            img_dtype = self.collection[i].pixel_array.dtype
            name = self.dcm_files[i]
            s += f"{i}) DCM of shape {img_shape}, dtype {img_dtype}\n"
            s += f"\t Filepath: {name}\n \n"
        s += "-----------------------------------------\n MASKS\n"
        for i in range(len(self.mask_files)):
            img_shape = self.masks[i].shape
            img_dtype = self.masks[i].dtype
            name = self.mask_files[i]
            s += f"{i}) Image of shape {img_shape}, dtype {img_dtype}\n"
            s += f"\t Filepath: {name}\n \n"
        return s

    def __repr__(self):
        return f"DcmSequence of {len(self.collection)} DICOMS, {len(self.masks)} Masks"

    def load_dcm(self, src: str) -> None:
        """
        Add a dicom to the collection
        :param src: Source directory to read the dicoms in from.
        :return: None
        """
        files = glob.glob(os.path.normpath(src + "/*.dcm"))
        if len(files) == 0:
            RuntimeWarning("No DICOM files found in this directory")
        else:
            for file in files:
                if file not in self.dcm_files:
                    ds = pydicom.dcmread(file)
                    self.collection.append(ds)
                    self.dcm_files.append(file)


    def save_dcm(self, dest: str) -> None:
        """
        Save the dicom to the destination folder.
        :param dest: destination folder
        :return: None
        """
        dest = os.path.normpath(dest + '/')
        for i, path in enumerate(self.dcm_files):
            filename = os.path.basename(path)
            self.collection[i].save_as(os.path.join(dest, filename))


    def load_mask(self, src: str) -> None:
        """
        Add a mask to the masks
        :param src: Source directory to read the masks in from.
        :return: None
        """
        files = glob.glob(os.path.normpath(src + "/*"))
        if len(files) == 0:
            RuntimeWarning("No DICOM files found in this directory")
        else:
            for file in files:
                if file not in self.mask_files:
                    img = cv2.imread(file, flags=0)
                    self.masks.append(img)
                self.mask_files.append(file)


    def save_mask(self, dest: str) -> None:
        """
        Save the masks to the destination folder.
        :param dest: destination folder
        :return: None
        """
        dest = os.path.normpath(dest + '/')
        for i, path in enumerate(self.mask_files):
            filename = os.path.basename(path)
            cv2.imwrite(os.path.join(dest, filename), self.masks[i])

    def is_empty(self):
        return len(self.collection) == 0 and len(self.masks) == 0

    def sort_dcm(self, reverse: bool = False) -> None:
        """
        Sort the dicoms based on their names in dcm_files.
        :param reverse: whether to sort in ascending or descending order.
                        False is ascending. True is descending.
        :return: None
        """
        zipped = zip(self.dcm_files, self.collection)
        x = sorted(zipped, reverse=reverse)
        self.dcm_files = [d for d, c in x]
        self.collection = [c for d, c in x]


    def sort_mask(self, reverse: bool = False) -> None:
        """
        Sort the masks based on their names in mask_files.
        :param reverse: whether to sort in ascending or descending order.
                        False is ascending. True is descending.
        :return: None
        """
        zipped = zip(self.mask_files, self.masks)
        x = sorted(zipped, reverse=reverse)
        self.mask_files = [d for d, c in x]
        self.masks = [c for d, c in x]


    def remove_dcm(self, **kwargs) -> None:
        """
        Remove a dicom and path from the collection.
        :param kwargs: Expecting to receive name or idx of file to remove
        :return: None
        """
        if 'name' not in kwargs.keys() and 'idx' not in kwargs.keys():
            raise KeyError('Expected either filename or index to delete')
        elif 'name' in kwargs.keys():
            name = kwargs['name']
            if name not in self.dcm_files:
                raise ValueError(name + ' is not in files')
            else:
                idx = self.dcm_files.index(name)
                self.dcm_files.remove(name)
                self.collection.remove(self.collection[idx])
        elif 'idx' in kwargs.keys():
            idx = kwargs['idx']
            if idx >= len(self.dcm_files):
                raise ValueError('Index out of bounds')
            else:
                self.dcm_files.remove(self.dcm_files[idx])
                self.collection.remove(self.collection[idx])


    def remove_mask(self, **kwargs) -> None:
        """
        Remove a mask and path from the collection.
        :param kwargs: Expecting to receive name or idx of file to remove
        :return: None
        """
        if 'name' not in kwargs.keys() and 'idx' not in kwargs.keys():
            raise KeyError('Expected either filename or index to delete')
        elif 'name' in kwargs.keys():
            name = kwargs['name']
            if name not in self.mask_files:
                raise ValueError(name + ' is not in files')
            else:
                idx = self.mask_files.index(name)
                self.mask_files.remove(name)
                self.masks.remove(self.masks[idx])
        elif 'idx' in kwargs.keys():
            idx = kwargs['idx']
            if idx >= len(self.mask_files):
                raise ValueError('Index out of bounds')
            else:
                self.mask_files.remove(self.mask_files[idx])
                self.masks.remove(self.masks[idx])


    def resize(self, dim: Tuple[int, int]) -> None:
        """
        Resize all dicoms and masks to the same dimension.
        :param dim: desired dimension of images (rows, columns)
        :return: None
        """
        for i in range(len(self.dcm_files)):
            ds = self.collection[i]
            image = ds.pixel_array
            downsampled = cv2.resize(image, dim)
            ds.PixelData = downsampled.tobytes()
            ds.Rows, ds.Columns = downsampled.shape

            self.collection[i] = ds

        for i in range(len(self.mask_files)):
            img = self.masks[i]
            img = cv2.resize(img, dim)
            img[img < 128] = 0
            img[img >= 128] = 255
            self.masks[i] = img


    def plot_norms(self, start: int = 0, end: Union[int, None] = None) -> None:
        """
        View the comparisons of preprocessing algorithms of dicom images in the collection
        from start to end indices. Press q to view next image.
        :param start: Which index to start at in the collection.
        :param end: Which index to stop at in the collection (exclusive). Value of None will plot
        every image.
        :return: None
        """

        if end is None:
            clip_lim = 40
            tile_grid_size = (8, 8)

            for i in range(len(self.collection)):
                img = self.collection[i].pixel_array
                img_cr = apply_cr_normalization(img)
                img_clahe = apply_clahe(img, clip_lim=clip_lim, tile_grid_size=tile_grid_size)
                img_fiji = apply_fiji_normalization(img)

                plot_comparisons(img, cr=img_cr, fiji=img_fiji, clahe=img_clahe)
                print('Press q to close this plot and view next image')
                plt.waitforbuttonpress()

        elif isinstance(end, int):
            clip_lim = 40
            tile_grid_size = (8, 8)

            for i in range(start, end):
                img = self.collection[i].pixel_array
                img_cr = apply_cr_normalization(img)
                img_clahe = apply_clahe(img, clip_lim=clip_lim, tile_grid_size=tile_grid_size)
                img_fiji = apply_fiji_normalization(img)

                plot_comparisons(img, cr=img_cr, fiji=img_fiji, clahe=img_clahe)
                print('Press q to close this plot and view next image')
                plt.waitforbuttonpress()


    def volshow(self, start: int = 0, end: Union[int, None] = None) -> None:
        """
        View the dicom images in the collection from start to end indices. Press left and
        right arrow keys to scroll through.
        :param start: Which index to start at in the collection.
        :param end: Which index to stop at in the collection (exclusive). Value of None
        will plot every image.
        :return: None
        """

        if end is None:
            vol = get_png(self.collection[start:])
            multi_slice_viewer(np.array(vol))

        elif isinstance(end, int):
            vol = get_png(self.collection[start:end])
            multi_slice_viewer(np.array(vol))


    def mask_show(self, start: int = 0, end: Union[int, None] = None) -> None:
        """
        View the mask images in the collection from start to end indices. Press left and
        right arrow keys to scroll through.
        :param start: Which index to start at in the collection.
        :param end: Which index to stop at in the collection (exclusive). Value of None
        will plot every image.
        :return: None
        """

        if end is None:
            multi_slice_viewer(np.array(self.masks))

        elif isinstance(end, int):
            multi_slice_viewer(np.array(self.masks[start:end]))


    def interpolate(self, interp_factor: float, clahe: bool = False, norm_alg: int = 1) -> np.ndarray:
        """
        Create an interpolated volume from the image stack. This will interpolate slices of
        images between every consecutive pair of slices. The num_slices determines how
        many interpolated slices are between the original slices and the separation between them.
        :param interp_factor: interpolation factor
        :param clahe: whether or not to perform clahe on the images
        :param norm_alg: which normalization algorithm to use to get the image between 0-255.
        If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
        normalization. norm_alg = 2 is for CR normalization.
        :return: the entire interpolated volume
        """
        stack = interpolate_dcm2vol(self.collection, interp_factor, clahe=clahe, norm_alg=norm_alg)
        return stack


    def interpolate_mask_volume(self, num_slices: int = 4, binarize: bool = False, split_val: int = 128) -> np.ndarray:
        """
        Create an interpolated volume from the image stack. This will interpolate slices of
        images between every consecutive pair of slices. The num_slices determines how
        many interpolated slices are between the original slices and the separation between them.
        :param num_slices: Number of interpolated slices between the original slices
        :param binarize: whether to binarize the interpolated volume or not
        :param split_val: int value to split the binarization
        :return: the entire interpolated volume
        """
        images = np.array(self.masks)
        stack = interpolate_imgs2vol(images, num_slices=num_slices)
        if binarize:
            stack[stack < split_val] = 0
            stack[stack >= split_val] = 255
        return stack


    def get_png(self, clahe: bool = False, norm_alg: int = 1) -> (List[str], List[np.ndarray]):
        """
        Get list of png images and list of file names by converting the current dicoms in the
        collection to 8-bit using the preferred norm-alg.
        :param clahe: whether or not to perform clahe on the images
        :param norm_alg: which normalization algorithm to use to get the image between 0-255.
        If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
        normalization. norm_alg = 2 is for CR normalization.
        :return: List of file names (not the path), list of png images
        """
        # use if statements for the 3 dif algorithms for normalization
        png_names = []
        for path in self.dcm_files:
            png_names.append(os.path.splitext(os.path.basename(path))[0] + '.png')

        images = get_png(self.collection, clahe=clahe, norm_alg=norm_alg)
        return png_names, images


    def convert_to_8bit(self, clahe: bool = False, norm_alg: int = 1) -> None:
        """
        Convert 16-bit dicoms to 8-bit dicoms.
        :clahe: whether or not to use the CLAHE algorithm on the image beforehand
        :norm_alg: which normalization algorithm to use to get the image between 0-255.
        If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
        normalization. norm_alg = 2 is for CR normalization.
        :return: None
        """
        for i, path in enumerate(self.dcm_files):
            ds = self.collection[i]

            if ds.pixel_array.dtype == np.uint16 or ds.pixel_array.dtype == np.int16:
                name = os.path.basename(path)
                new_ds = _get_new_ds(ds, name)
                image = ds.pixel_array
                image = convert_int_to_uint(image)
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

                new_ds.PixelData = image.tobytes()

                self.collection[i] = new_ds
