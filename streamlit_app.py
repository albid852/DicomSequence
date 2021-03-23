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
from PlotDCM import plot_comp2fiji
from DCMSequence import DcmSequence


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


st.set_page_config("Pydicom Image")
st.title("Pydicom Image")

dcm_list = []
img_list = []
dcms = DcmSequence()

# password
password = "Raveen is Notion Ambassador"
password_text = st.text_input('Input Password', type='password')

# I don't think there is way to remove the password box after the correct password as been inputed due to the
# limitations of streamlit
if password != password_text:
    if password_text:
        st.text("Sorry Incorrect Password")
else:
    uploaded_file = st.file_uploader("Upload Files", accept_multiple_files=True, type='dcm')
    if len(uploaded_file) > 0:
        for file in uploaded_file:
            dcm_path = DicomBytesIO(file.read())
            ds = pydicom.dcmread(dcm_path)
            dcms.collection.append(ds)
            dcms.dcm_files.append(file.name)

        filenames, img_list = dcms.get_png()

        if len(img_list) > 1:
            st.sidebar.title("Set Image Size")
            display = ("Size", "Height, Width")
            options = list(range(len(display)))
            hw_check = st.sidebar.selectbox("Set Height and Width or Size?",
                                            options=options,
                                            format_func=lambda x: display[x])
            if hw_check:
                height_slide = st.sidebar.slider("Set Width", 1, 512, value=img_list[0].shape[1])
                width_slide = st.sidebar.slider("Set Height", 1, 512, value=img_list[0].shape[0])
                img_list = resize(img_list, (height_slide, width_slide))
            else:
                size_slide = st.sidebar.slider("Set Image Size", 1, 512, value=img_list[0].shape[0])
                img_list = resize(img_list, (size_slide, size_slide))

            st.sidebar.title("Sort DICOM's by File Name")
            sort = st.sidebar.checkbox("Sort by file name")
            if sort:
                dcms.sort_dcm()
                dcms.sort_mask()

            slide = st.slider("Dicom Image", 1, len(img_list))
            st.image(img_list[slide - 1])  # can use css to center this

            name = st.text(f"File name: {filenames[slide - 1]}")

            norm_check = st.checkbox("Visualize normalizations")

            if norm_check:
                st.sidebar.title("Apply normalizations to Images")
                st.pyplot(plot_comp2fiji(img_list[slide - 1]))
                clahe_check = st.sidebar.checkbox("Apply CLAHE")
                cr_check = st.sidebar.checkbox("Apply CR normalization")

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