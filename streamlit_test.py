import streamlit as st
import numpy as np
import pydicom
from pydicom.filebase import DicomBytesIO
from scipy import interpolate
import cv2
import base64
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image
from io import BytesIO

st.set_page_config("Pydicom Image")
st.title("Pydicom Image")

dcm_list = []


def convert_int_to_uint(img):
    """
    Conversion of int16 to uint16
    :param img: numpy array to convert
    :return: numpy array as type uint16
    """
    if img.dtype == np.int16:
        img_min = np.min(img)
        img += abs(img_min)
        return img.astype(np.uint16)


def apply_clahe(img, clip_lim=40, tile_grid_size=(8, 8)):
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


def apply_fiji_normalization(img):
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


def apply_cr_normalization(img):
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


def get_png(dcm_list, clahe=False, norm_alg=1):
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
    images = []

    for ds in dcm_list:

        if ds.pixel_array.dtype == np.uint8:
            images.append(ds.pixel_array)

        elif ds.pixel_array.dtype == np.uint16 \
                or ds.pixel_array.dtype == np.int16:
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

            images.append(image)

    return images

@st.cache(suppress_st_warning=True, show_spinner=False)
def interpolate_volume(vol, num_slices=4, clahe=False, norm_alg=1):
    """
    Create an interpolated volume from the image stack. This will interpolate slices of
    images between every consecutive pair of slices. The num_slices determines how
    many interpolated slices are between the original slices and the separation between them.
    :param num_slices: Number of interpolated slices between the original slices
    :param clahe: whether or not to perform clahe on the images
    :param norm_alg: which normalization algorithm to use to get the image between 0-255.
    If using clahe, recommended to set norm_alg = 0. norm_alg = 1 is for the fiji
    normalization. norm_alg = 2 is for CR normalization.
    :return: the entire interpolated volume
    """
    images = get_png(vol, clahe=clahe, norm_alg=norm_alg)
    volume = np.array(images)

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


def get_image_download_link(vol):
    """
    Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "a", ZIP_DEFLATED, False) as zip_file:
        for i in range(len(vol)):
            img_buffer = BytesIO()
            result = Image.fromarray(vol[i])
            result.save(img_buffer, format="PNG")
            zip_file.writestr(f"{i}.png", img_buffer.getvalue())

    result_str = base64.b64encode(zip_buffer.getvalue()).decode()
    href = f'<a href="data:file/zip;base64,{result_str}" download="interpolated.zip">' \
           f'Download result</a>'
    return href


uploaded_file = st.file_uploader("Upload Files", accept_multiple_files=True, type='dcm')
if len(uploaded_file) > 0:
    for file in uploaded_file:
        raw = DicomBytesIO(file.read())
        ds = pydicom.dcmread(raw)
        dcm_list.append(ds)

    slide = st.slider("Dicom Image", 1, len(dcm_list))
    st.image(dcm_list[slide - 1].pixel_array)  # can use css to center this

    interp_check = st.checkbox("Interpolate Volume")
    if interp_check:
        st.header("Interpolation")
        x = st.number_input("Number of slices between", 0, 35)
        with st.spinner("Interpolating..."):
            dcm_vol = interpolate_volume(dcm_list, num_slices=x)
        st.success("Successfully Interpolated!")
        slide2 = st.slider("Interpolated Image", 1, len(dcm_vol))
        st.image(dcm_vol[slide2 - 1])  # can use css to center this
        if st.button("Download"):
            with st.spinner("Getting your link ready..."):
                st.markdown(get_image_download_link(dcm_vol), unsafe_allow_html=True)
