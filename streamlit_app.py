import streamlit as st
import numpy as np
import pydicom
from pydicom.filebase import DicomBytesIO
import base64
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image
from io import BytesIO
from scipy import interpolate
from preprocessing import get_png, resize
    # convert_to_8bit


@st.cache(suppress_st_warning=True, show_spinner=False)
def interpolate_volume(volume: np.ndarray, num_slices: int = 4) -> np.ndarray:
    """
    Create an interpolated volume from the image stack. This will interpolate slices of
    images between every consecutive pair of slices. The num_slices determines how
    many interpolated slices are between the original slices and the separation between them.
    :param volume: array of images to interpolate between
    :param num_slices: Number of interpolated slices between the original slices
    :return: the entire interpolated volume
    """
    if num_slices <= 0:
        return volume
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


def get_image_download_link(vol: np.ndarray) -> str:
    """
    Generates a link that allows the png image or volume to be downloaded in a zip file
    :param vol: numpy array of png images
    :return: href string
    """
    if len(vol.shape) == 2:
        vol = np.expand_dims(vol, axis=0)
    # are there security concerns for the future?
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


def plot_norms(img: np.ndarray) -> None:
    pass



st.set_page_config("Pydicom Image")
st.title("Pydicom Image")

dcm_list = []
img_list = []

uploaded_file = st.file_uploader("Upload Files", accept_multiple_files=True, type='dcm')
if len(uploaded_file) > 0:
    for file in uploaded_file:
        dcm_path = DicomBytesIO(file.read())
        ds = pydicom.dcmread(dcm_path)
        dcm_list.append(ds)

    img_list = get_png(dcm_list)

    if st.button("Clear All"):
        uploaded_file = []

    if len(img_list) > 1:
        slide = st.slider("Dicom Image", 1, len(img_list))
        st.image(img_list[slide - 1])  # can use css to center this

        interp_check = st.checkbox("Interpolate Volume")
        if interp_check:
            st.header("Interpolation")
            n = st.number_input("Number of slices between", 0, 25)

            with st.spinner("Interpolating..."):
                dcm_vol = interpolate_volume(np.array(img_list), num_slices=n)
            st.success("Successfully Interpolated!")

            slide2 = st.slider("Interpolated Image", 1, len(dcm_vol))
            st.image(dcm_vol[slide2 - 1])  # can use css to center this

            if st.button("Download"):
                with st.spinner("Getting your link ready..."):
                    st.markdown(get_image_download_link(dcm_vol), unsafe_allow_html=True)
                st.success("Click the link to download your images!")
    else:
        st.image(img_list[0])
