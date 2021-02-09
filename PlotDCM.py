import numpy as np
import matplotlib.pyplot as plt
from preprocessing import apply_clahe, apply_cr_normalization, apply_fiji_normalization
from typing import Union


# SLICE VIEWER
def multi_slice_viewer(volume: np.ndarray):
    """
    Go slice-by-slice through a sequence of images stacked as a volume. Recommended to
    use for viewing the output of DcmSequence.interpolate_volume()
    :param volume: Stack of images
    :return: None
    """
    def process_key(event):
        f = event.canvas.figure
        a = f.axes[0]
        if event.key == 'left':
            previous_slice(a)
        elif event.key == 'right':
            next_slice(a)
        a.set_title(str(a.index))
        f.canvas.draw()

    def previous_slice(a):
        vol = a.volume
        a.index = (a.index - 1) % vol.shape[0]  # wrap around using %
        a.images[0].set_array(vol[a.index])

    def next_slice(a):
        vol = a.volume
        a.index = (a.index + 1) % vol.shape[0]
        a.images[0].set_array(vol[a.index])

    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 0
    ax.set_title(str(ax.index))
    ax.imshow(volume[ax.index], cmap='gray')
    print("Use the right arrow key to go to the next slice")
    print("Use the left arrow key to go to the previous slice")
    fig.canvas.mpl_connect('key_press_event', process_key)


# PLOTTING PROCESSING COMPARISONS
def plot_comparisons(original: np.ndarray, cr: Union[np.ndarray, None] = None,
                     fiji: Union[np.ndarray, None] = None,
                     clahe: Union[np.ndarray, None] = None) -> plt.Figure:
    """
    Visualization of the different image processing algorithms in a 2x2 grid using
    matplotlib.
    :param original: original image
    :param cr: cr processed version of the image
    :param fiji: fiji processed version of the image
    :param clahe: clahe processed version of the image
    :return: plotted figure
    """
    fig = plt.figure()
    plt.axis('off')
    plt.tick_params(axis='both')
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('Original')
    ax2.title.set_text('CR')
    ax3.title.set_text('FIJI')
    ax4.title.set_text('CLAHE')

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    if cr is None:
        cr = apply_cr_normalization(original)
    if clahe is None:
        clahe = apply_clahe(original)
    if fiji is None:
        fiji = apply_fiji_normalization(original)

    ax1.imshow(original, cmap='gray')
    ax2.imshow(cr, cmap='gray')
    ax3.imshow(fiji, cmap='gray')
    ax4.imshow(clahe, cmap='gray')

    # plt.show()
    return fig

def plot_comp2fiji(img: np.ndarray,
                   cr: Union[np.ndarray, None] = None,
                   clahe: Union[np.ndarray, None] = None) -> plt.Figure:
    """
        Visualization of the different image processing algorithms in a 2x2 grid using
        matplotlib.
        :param img: the 8-bit image or the fiji processed version of the image
        :param cr: cr processed version of the image
        :param clahe: clahe processed version of the image
        :return: plotted figure
        """
    fig = plt.figure()
    plt.axis('off')
    plt.tick_params(axis='both')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.title.set_text('Original')  # 8-bit image / fiji version
    ax2.title.set_text('CR')
    ax3.title.set_text('CLAHE')

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    if cr is None:
        cr = apply_cr_normalization(img)
    if clahe is None:
        clahe = apply_clahe(img)

    ax1.imshow(img, cmap='gray')
    ax2.imshow(cr, cmap='gray')
    ax3.imshow(clahe, cmap='gray')
    return fig