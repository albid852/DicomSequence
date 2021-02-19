# import streamlit as st
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import meshio
from PlotDCM import multi_slice_viewer
from PIL import Image

# # st.title("Pointcloud Visuals")

def count_nonzero(vol: np.ndarray) -> int:
    c = 0
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                if vol[i, j, k].all() != 0:
                    c += 1
    return c

# try just appending to rgb and points (all in one go) instead of counting then indexing
def vol2points(vol: np.ndarray) -> np.ndarray:
    cpixels = count_nonzero(vol)
    rgb = np.zeros((cpixels, 3), dtype=np.uint8)
    points = np.zeros((cpixels, 3), dtype=np.int16)
    n = 0
    for i in range(vol.shape[0]):  # z
        for j in range(vol.shape[1]):  # y
            for k in range(vol.shape[2]):  # x
                if vol[i, j, k].all() != 0:
                    points[n] = [k, j, i]
                    rgb[n] = vol[i, j, k, :3]
                    n += 1
    return rgb, points


def points2vol(points: np.ndarray, rgb: np.ndarray, yx_shape=(256, 256)) -> np.ndarray:
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    points[:, 2] += abs(np.min(points[:, 2]))  # making sure z are positive
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    vol = np.zeros((z_max + 50, y_max + 50, x_max + 50, 4), dtype=np.uint8)

    c = 0
    for x, y, z in points:
        vol[z, y, x] = np.append(rgb[c], 255)
        c += 1

    return vol
    tmp = np.zeros((vol.shape[0], yx_shape[0], yx_shape[1], 4), dtype=np.uint8)
    for i in range(vol.shape[0]):
        img = Image.fromarray(vol[i], 'RGBA')
        img = img.resize(yx_shape)
        tmp[i] = np.asarray(img)
    return tmp

mesh = meshio.read('/Users/albi/Downloads/sample.ply')
xyz = mesh.points.astype(np.int16)
rgb = np.array(list(mesh.point_data.values())).T

np.unique(xyz[:, 2])

vol = points2vol(xyz, rgb, yx_shape=(256, 256))  # downsample and conversion works

# NEXT is to convert from volume to points
new_rgb, new_xyz = vol2points(vol)

multi_slice_viewer(vol)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_axis_off()
# ax.scatter(xyz[:, 2], xyz[:, 1], xyz[:, 0], c=rgb[:]/255., s=0.1)
# st.pyplot(fig)





