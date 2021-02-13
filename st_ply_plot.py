# import streamlit as st
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import meshio
from skimage.measure import block_reduce
#
# st.title("Pointcloud Visuals")
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_axis_off()

mesh = meshio.read('/Users/albi/Downloads/sample.ply')
xyz = mesh.points.astype(np.int16)
rgb = np.array(list(mesh.point_data.values())).T

x_max = np.max(xyz[:, 0])
y_max = np.max(xyz[:, 1])
z_min, z_max = np.min(xyz[:, 2]), np.max(xyz[:, 2])

vol = np.zeros((x_max + 50, y_max + 50, abs(z_min) + 50, 4), dtype='uint8')

xyz[:, 2] += abs(z_min)

for i, (x, y, z) in enumerate(xyz):
    vol[x, y, z] = np.append(rgb[i], 1)

aaa = block_reduce(vol, (2, 2, 1, 1), np.mean, func_kwargs={'dtype': np.uint8})
# works, but reduces WAY too much, to the point where like nothing is real. Would have to resample values if anything
# multiplying by scalar (no addition to leave 0's) to increase intensity in RGB would work wonders (don't hit Alpha)

# ax.scatter(xyz[-100000:, 0], xyz[-100000:, 1], xyz[-100000:, 2], c=rgb.T[-100000:]/255., s=0.1)

# st.pyplot(fig)





