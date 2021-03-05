# import streamlit as st
import numpy as np
from scipy import interpolate
from scipy.ndimage import zoom
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
def vol2points(vol: np.ndarray, min_val: int = 0) -> np.ndarray:
    cpixels = count_nonzero(vol)
    print(f"Number of non-black pixels: {cpixels}")
    rgb = np.zeros((cpixels, 3), dtype=np.uint8)
    points = np.zeros((cpixels, 3), dtype=np.int16)
    n = 0
    for i in range(vol.shape[0]):  # z
        for j in range(vol.shape[1]):  # y
            for k in range(vol.shape[2]):  # x
                if n % 50000 == 0:
                    print(f"{n} points written")
                if vol[i, j, k] > min_val:
                    points[n] = [k, j, i]
                    rgb[n] = vol[i, j, k, :3]
                    n += 1
    print(f"{n} points written")
    return rgb, points


def points2vol(points: np.ndarray, rgb: np.ndarray) -> np.ndarray:
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

vol = points2vol(xyz, rgb)  # conversion works

new_array = zoom(vol, (0.5, 0.5, 0.5, 1))  # downsample = zooming in and out
# multi_slice_viewer(vol)
# multi_slice_viewer(new_array)

vol_min = np.min(rgb[:, 0])  # assuming rgb are all the same (grayscale image)
new_rgb, new_xyz = vol2points(vol, min_val=vol_min)
new_rgb_resized, new_xyz_resized = vol2points(new_array, min_val=vol_min)

# Now write a new mesh with the RESIZED one



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_axis_off()
# Problem: Surface of mesh coloring is very messed up from the zoom. Probably can't fix it.
ax.scatter(new_xyz_resized[:, 2], new_xyz_resized[:, 1], new_xyz_resized[:, 0], c=new_rgb_resized[:]/255., s=0.1)
ax.scatter(new_xyz[:, 2], new_xyz[:, 1], new_xyz[:, 0], c=new_rgb[:]/255., s=0.1)

# st.pyplot(fig)





