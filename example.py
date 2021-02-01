from DCMSequence import DcmSequence, multi_slice_viewer

dcms = DcmSequence()


# LOADING DICOMS
# feed in all dicoms from the path directory
path = "../dicom-examples/series-000006/"
dcms.load_dcm(path)



# REMOVING DICOMS
# remove some dicoms that we do not want either by specifying the dicom name, or index
dcms.remove_dcm(idx=0)

dicom_name = dcms.dcm_files[4]
dcms.remove_dcm(name=dicom_name)

# can remove more than 1 using a for loop
for i in range(3):
    dcms.remove_dcm(idx=0)

    

# RESIZING DICOMS
# can resize all the dicoms to any shape by specifying a tuple as a dimension
dcms.resize((256, 256))



# GETTING PNG IMAGES
# can use 3 different algorithms to get the png images, as well as using CV2's CLAHE algorithm
# to locally normalize an image
linear_names, linear_images = dcms.get_png(norm_alg=1)  # linear normalization
cr_names, cr_images = dcms.get_png(norm_alg=2)  # cr normalization
clahe_names, clahe_images = dcms.get_png(clahe=True, norm_alg=0,
                                         clip_lim=40, tile_grid_size=(8, 8))  # CLAHE normalization



# VISUALIZE NORMALIZATIONS
# before making these transformations, you can view comparisons between the images using matplotlib
# for as many of the images as you want. Press q to close the current plot and open the next one
dcms.plot_norms(start=1, end=3)



# VISUALIZE ALL DICOMS
# press left and right arrow keys to scroll through
dcms.volshow(start=1, end=3)



# CONVERT TO 8-BIT
# this will convert ALL of the dicoms in the collection to 8-bit using whichever normalization
# algorithm you choose. This is NOT currently reversible so be sure of which you want to use.
dcms.convert_to_8bit(clahe=True, norm_alg=1)



# INTERPOLATE VOLUME
# to interpolate images between the slices you have
# and you can visualize it slice-by-slice (3D rendering after interpolation coming soon)
volume = dcms.interpolate_volume(num_slices=20, clahe=False, norm_alg=1)
multi_slice_viewer(volume)



# SAVE THE DICOMS
# this will save all dicoms in the collection to a specified directory with the same name that was
# given.
destination = "../new_folder/"
dcms.save_dcm(destination)


