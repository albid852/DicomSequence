# import streamlit as st
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import meshio
from skimage.measure import block_reduce
from PlotDCM import multi_slice_viewer
from skimage.transform import resize
from PIL import Image

# # st.title("Pointcloud Visuals")

def vol2points(vol: np.ndarray) -> np.ndarray:
    pass

def points2vol(points: np.ndarray, rgb: np.ndarray, yx_shape=(256, 256), dtype: str = 'uint8') -> np.ndarray:
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    points[:, 2] += abs(np.min(points[:, 2]))  # making sure z are positive
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    vol = np.zeros((z_max + 50, y_max + 50, x_max + 50, 4), dtype=dtype)

    c = 0
    for x, y, z in points:
        vol[z, y, x] = np.append(rgb[c], 1)
        c += 1

    tmp = np.zeros((vol.shape[0], yx_shape[0], yx_shape[1], 4), dtype=dtype)
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


# multi_slice_viewer(vol[:, :, :, :3])
#
# abc = rgb[-100000:]/255.

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_axis_off()
# ax.scatter(xyz[:, 2], xyz[:, 1], xyz[:, 0], c=rgb[:]/255., s=0.1)
# st.pyplot(fig)





