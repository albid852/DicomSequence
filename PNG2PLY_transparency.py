from DCMSequence import DcmSequence
import argparse
import os
import img2ply
import open3d as o3d
import numpy as np
from img2ply import generate_ply

if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description="Provide path of DICOM images and interpolation factor")
    parser.add_argument('path', metavar='p', type=str,
                        help='The directory or file path of the DICOM images')
    parser.add_argument('interp_factor', metavar='f', type=float,
                        help='Determines number of slices when using method 0')
    args = parser.parse_args()
    path = args.path
    interp_factor = args.interp_factor
    if interp_factor < 0 or interp_factor > 1:
        RuntimeWarning(f"Interp_factor received: {interp_factor}, but interp_factor must be between 0 and 1. "
                       f"Changed to 0.1")
        interp_factor = 0.1

    print("***** LOADING DICOMS *****")

    # build DcmSequence and load DICOMS
    dcms = DcmSequence()
    dcms.load_dcm(path)
    images = dcms.interpolate(interp_factor)

    print("***** BEGINNING PLY CONVERSION *****")
    ply_path = generate_ply(images, path)

    print("***** VISUALIZNG PLY *****")
    # Visualizing Point Cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)  # downsample

    # drawing both results
    # o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd])