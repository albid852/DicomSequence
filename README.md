# DICOM Sequence
Using the DcmSequence class in the DCMSequence.py file you can load in DICOM directories, single files, or masks for segmentation purposes and perform the following operations on the images:
 - Resize 
 - Normalize to 0-255 range using any of 3 different algorithms
 - Plot comparisons between these 3 different algorithms and choose which you want to use 
 - Convert 16-bit DICOM to 8-bit DICOM 
 - Create a new DICOM header based on some existing dataset 
 - Get png images of all DICOM's 
 - Interpolate a volume from loaded slices
 - View these many different slices using the multi-slice viewer

Dependencies: Numpy, Scipy, OpenCV, PyDicom, Matplotlib, Streamlit, PIL

# Example Usage

    from DCMSequence import DcmSequence
    
    dcms = DcmSequence()
    dcms.load_dcm(path/to/dicoms)
    dcms.resize((256, 256))
    dcms.convert_to_8bit()
    dcms.save_dcm(path/to/save)
For more detailed examples, go to example.py which has a detailed guide on how to use this collection.

# Streamlit
Streamlit is a data-oriented UI application that allows you to create your own webpage. We utilize it for our own frontend to make it much easier to get png images from your DICOM's as well as get an interpolated volume from the png images. 

