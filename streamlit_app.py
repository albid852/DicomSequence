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
from DCMSequence import DcmSequence
from typing import List
import math
from img2ply import generate_ply

@st.cache(suppress_st_warning=True, show_spinner=False)
def interpolate_volume(dcm_list: List[pydicom.dataset.FileDataset],
                       img_list: List[np.ndarray],
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
    rgi = interpolate.RegularGridInterpolator(points, img_list)

    # check whether to use Spacing between slices, Slice Location or neither
    method = None  # 0 = slice location,  1 = spacing between slices,  2 = not given
    prev = None
    n = 0
    while n < depth:
        d = dcm_list[n]
        if hasattr(d, "SpacingBetweenSlices"):
            if method == 0 or method == 2:
                AttributeError(f"""DICOM {n} does not utilize Spacing Between Slices. 
                                 All DICOMS must use the same positioning technique""")
            if prev is not None and prev != d.SpacingBetweenSlices:
                AttributeError(f"""DICOM {n} using Spacing Between Slices does not have the 
                                 same spacing as previous ones.""")
            elif prev is None:
                prev = d.SpacingBetweenSlices
            method = 1
        elif hasattr(d, "SliceLocation"):
            if method == 1 or method == 2:
                AttributeError(f"""DICOM {n} does not utilize Slice Location. 
                                 All DICOMS must use the same positioning technique""")
            method = 0
        else:
            if method == 0:
                AttributeError(f"""DICOM {n} utilizes Slice Location, but previous ones did not. 
                                 All DICOMS must use the same positioning technique""")
            if method == 1:
                AttributeError(f"""DICOM {n} utilizes Spacing Between Slices, but previous ones did not. 
                                 All DICOMS must use the same positioning technique""")

            method = 2
        n += 1

    if method == 2:
        # Conduct interpolation with constant 4 slices between
        # get slices with desired separation
        g = np.mgrid[1:5, :img_height, :img_width]  # 4 slices between (5 - 1)
        coords = np.vstack(map(np.ravel, g)).transpose().astype(np.float16)  # [z, y, x] between 2 slices
        coords[:, 0] *= 1 / 5  # scale to be properly between the slices

        # final interpolated volume but empty
        stack = np.zeros((depth + 4 * (depth - 1), img_height, img_width), dtype=np.uint8)

        for i in range(depth):
            stack[i * 5] = img_list[i]
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
            stack[i * (num_slices + 1)] = img_list[i]
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

                stack[elapsed] = img_list[i]
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

@st.cache(suppress_st_warning=True, show_spinner=False)
def threshold_images(imgs: List[np.ndarray], threshold: int):
    """
    Whiten pixels in the grayscale images by applying a threshold. All pixels below
    threshold intensity will become white
    :param imgs: list of images
    :param threshold: threshold value
    """
    img_arr = np.array(imgs)
    new_imgs = img_arr.copy()
    for i in range(len(imgs)):
        new_imgs[i][np.where(new_imgs[i] < threshold)] = 255
    return new_imgs

@st.cache(suppress_st_warning=True, show_spinner=False)
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
    href = f'<a href="data:file/zip;base64,{result_str}" download="dicoms.zip">' \
           f'Download result</a>'
    return href


st.set_page_config("Pydicom Image")
st.title("Pydicom Image")

# password
password = "BelugaWhales227"
password_text = st.text_input('Input Password', type='password')

if password != password_text:
    if password_text:
        st.warning("Incorrect Password")

else:
    dcms = DcmSequence()
    # upload files
    uploaded_file = st.file_uploader("Upload Files", accept_multiple_files=True, type='dcm')
    if len(uploaded_file) > 0:
        for file in uploaded_file:
            dcm_path = DicomBytesIO(file.read())
            ds = pydicom.dcmread(dcm_path)
            dcms.collection.append(ds)
            dcms.dcm_files.append(file.name)
        # option to sort by file name only if we got more than 1 image
        sortcol1, sortcol2 = st.beta_columns((2, 7))
        sort = sortcol1.button("Sort by file name")
        if sort:
            dcms.sort_dcm()
            dcms.sort_mask()
            sortcol2.success("DICOMS sorted!")

        # get png images
        filenames, img_list = dcms.get_png()
        display_imgs = img_list

        # set size of images with these sidebar sliders
        st.sidebar.title("Set Image Size")
        display = ("Size", "Height, Width")
        options = list(range(len(display)))
        hw_check = st.sidebar.selectbox("Set Height and Width or Size?",
                                        options=options,
                                        format_func=lambda x: display[x])
        col_width = 3
        if hw_check:
            height_slide = st.sidebar.slider("Set Width", 1, 512, value=img_list[0].shape[1])
            width_slide = st.sidebar.slider("Set Height", 1, 512, value=img_list[0].shape[0])
            img_list = resize(img_list, (height_slide, width_slide))
            dcms.resize((width_slide, height_slide))
            if width_slide >= 450:
                col_width = 3
            elif width_slide >= 350:
                col_width = 2
            else:
                col_width = 1

        else:
            size_slide = st.sidebar.slider("Set Image Size", 1, 512, value=img_list[0].shape[0])
            img_list = resize(img_list, (size_slide, size_slide))
            dcms.resize((size_slide, size_slide))
            if size_slide >= 450:
                col_width = 3
            elif size_slide >= 350:
                col_width = 2
            else:
                col_width = 1

        if len(img_list) > 1:
            # scroll through png images (not interpolated)
            st.header("Images")
            imgcol1, imgcol2 = st.beta_columns((col_width, 1))

            slide = imgcol2.slider("Dicom Image", 1, len(img_list))
            thresh_slide = imgcol2.slider("Threshold Value", 0, 255)

            if thresh_slide > 0:
                thresh_img_list = threshold_images(img_list, thresh_slide)
                display_imgs = thresh_img_list
            else:
                display_imgs = img_list

            imgcol1.image(display_imgs[slide - 1])
            name = imgcol1.text(f"File name: {filenames[slide - 1]}")

            # interpolate volume
            st.header("Interpolation")
            interpcol1, interpcol2 = st.beta_columns((col_width, 1))

            n = interpcol2.slider("Interpolation factor",
                                  0., 1., value=0., step=0.01)

            with st.spinner("Interpolating..."):
                dcm_vol = interpolate_volume(dcms.collection, display_imgs,
                                             interp_factor=n)

            slide2 = interpcol2.slider("Interpolated Image", 1, len(dcm_vol))

            interpcol1.image(dcm_vol[slide2 - 1])

            if st.button("Download"):
                with st.spinner("Getting your link ready..."):
                    st.markdown(get_image_download_link(dcm_vol), unsafe_allow_html=True)
                st.success("Click the link to download your images!")

            # ply_check = st.checkbox("Generate Point Cloud")


        else:
            # scroll through png images (not interpolated)
            st.header("Images")
            imgcol1, imgcol2 = st.beta_columns((col_width, 1))

            thresh_slide = imgcol2.slider("Threshold Value", 0, 255)

            if thresh_slide > 0:
                thresh_img_list = threshold_images(img_list, thresh_slide)
                display_imgs = thresh_img_list
            else:
                display_imgs = img_list

            imgcol1.image(display_imgs[0])
            name = imgcol1.text(f"File name: {filenames[0]}")

            if st.button("Download"):
                with st.spinner("Getting your link ready..."):
                    st.markdown(get_image_download_link(np.array(display_imgs)), unsafe_allow_html=True)
                st.success("Click the link to download your images!")
