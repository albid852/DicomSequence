from DCMSequence import DcmSequence
import argparse
import os
import img2ply
import open3d as o3d

if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description="Provide path of DICOM images")
    parser.add_argument('path', metavar='p', type=str,
                        help='the directory or file path of the DICOM images')
    args = parser.parse_args()
    path = args.path

    print("***** LOADING DICOMS *****")

    # build DcmSequence and load DICOMS
    dcms = DcmSequence()
    dcms.load_dcm(path)
    images = dcms.interpolate_dcm_volume(num_slices=4)

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

    #
    print("***** VISUALIZNG PLY *****")
    # Visualizing Point Cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])