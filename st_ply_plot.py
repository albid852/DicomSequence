import streamlit as st
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import meshio
from skimage.measure import block_reduce


# def downsample(pcloud_np, resolution):
#     xy = pcloud_np.T[:2]
#     xy = ((xy + resolution / 2) // resolution).astype(int)
#     mn, mx = xy.min(axis=1), xy.max(axis=1)
#     sz = mx + 1 - mn
#     flatidx = np.ravel_multi_index(xy-mn[:, None], sz)
#     histo = np.bincount(flatidx, pcloud_np[:, 2], sz.prod()) / np.maximum(1, np.bincount(flatidx, None, sz.prod()))
#     return histo.reshape(sz), *(xy * resolution)
#

st.title("Pointcloud Visuals")
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_axis_off()

mesh = meshio.read('/Users/albi/Downloads/sample.ply')
xyz = mesh.points
rgb = np.array(list(mesh.point_data.values()))

aaa = block_reduce(xyz, (30, 30), np.mean, func_kwargs={'dtype': np.uint8})

ax.scatter(xyz[-100000:, 0], xyz[-100000:, 1], xyz[-100000:, 2], c=rgb.T[-100000:]/255., s=0.1)

st.pyplot(fig)





