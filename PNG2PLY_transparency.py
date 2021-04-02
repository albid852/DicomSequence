from DCMSequence import DcmSequence
import argparse
import os
import img2ply
import open3d as o3d

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
        RuntimeError(f"Interp_factor received: {interp_factor}, but interp_factor must be between 0 and 1 inclusive")

    print("***** LOADING DICOMS *****")

    # build DcmSequence and load DICOMS
    dcms = DcmSequence()
    dcms.load_dcm(path)
    images = dcms.interpolate(interp_factor)

    print("***** BEGINNING PLY CONVERSION *****")

    # Giving writing permissions
    os.chmod(path, 0o777)

    ply_dir = os.path.join(path, "DCMPLY")
    if not os.path.exists(ply_dir):
        os.mkdir(ply_dir)
    ply_path = os.path.join(ply_dir, "sample.ply")

    # get dimensions of PLY (assuming all the same height and width)
    depth = len(images)
    height = images[0].shape[0]
    width = images[0].shape[1]
    print(f"Dimensions (h, w, d) of Bounding Box: "
          f"({height}, {width}, {depth})")

    # convert
    # THE BOUNDING BOX has to be [Width, Height, Number of Slices]
    img2ply.convert(
        images,
        ply_path,
        [width, height, 273],
        # [width, height, SliceNumber],
        direction="z",
        inverse=True,
        ignore_alpha=True,
        w_samples=0,
        h_samples=0,
        reduce_factor=5,
        maintain_aspect_ratio=True
    )

    print("***** VISUALIZNG PLY *****")
    # Visualizing Point Cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)  # downsample

    # drawing both results
    # o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([voxel_down_pcd])