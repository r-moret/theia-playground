import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def plot_mri(file_path: str, selected_slice: int, *, segmentation_path: str = None):

    img = nib.load(f"{file_path}").get_fdata()
    img_slice = img[:, :, selected_slice]

    if segmentation_path:
        label = nib.load(f"{segmentation_path}").get_fdata()
        label_slice = label[:, :, selected_slice]

        label_masked = np.ma.masked_where(label_slice == 0, label_slice)

        plt.imshow(img_slice.T, cmap="gray")
        plt.imshow(label_masked.T, cmap="gnuplot", alpha=0.5)
    else:
        plt.imshow(img_slice.T, cmap="gray")

    plt.show()
