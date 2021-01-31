Using the DcmSequence class in the DCMSequence.py file you can load in dicom directories or single files and perform the following 
operations on the images:

    Resize
    Normalize to 0-255 range using any of 3 different algorithms
    Plot comparisons between these 3 different algorithms and choose which you want to use
    Convert 16-bit dicoms to 8-bit dicoms
    Create a new dicom header based on some existing dataset
    Get png images of the dicom's pixel arrays
    Interpolate a volume composed of many different slices from your dicom collection
    View these many different slices using a multi-slice viewer functionality
    
And you can choose to save the dicoms using the exact same name to some new directory.

Dependencies:
    numpy
    pydicom
    opencv
    matplotlib
    scipy
    
