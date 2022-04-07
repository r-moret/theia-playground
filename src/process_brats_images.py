import os
import shutil

import cv2 as cv
import nibabel as nib
import numpy as np

from typing import Tuple
from tqdm import tqdm


def resize_images(
    originals_folder: str,
    labels_folder: str,
    new_size: Tuple[int, int],
    *,
    mode: int = cv.IMREAD_GRAYSCALE,
):
    print("Resizing...")
    for image_name in tqdm(os.listdir(originals_folder), ncols=70):
        img = cv.imread(f"{originals_folder}/{image_name}", mode)
        label = cv.imread(f"{labels_folder}/{image_name}", mode)

        resized_img = cv.resize(img, new_size, interpolation=cv.INTER_NEAREST)
        resized_label = cv.resize(label, new_size, interpolation=cv.INTER_NEAREST)

        cv.imwrite(f"{originals_folder}/{image_name}", resized_img)
        cv.imwrite(f"{labels_folder}/{image_name}", resized_label)


def move_and_rename_images(
    old_folder: str,
    new_folder: str,
    *,
    original_file_termination: str = "_t1.nii",
    label_file_termination: str = "_seg.nii",
):
    print("Moving...")
    for img_counter, folder in enumerate(tqdm(os.listdir(old_folder), ncols=70)):
        files = os.listdir(f"{old_folder}/{folder}")

        original_file = [
            original
            for original in files
            if original[-len(original_file_termination) :] == original_file_termination
        ][0]
        segmentation_file = [
            segmentation
            for segmentation in files
            if segmentation[-len(label_file_termination) :] == label_file_termination
        ][0]

        os.makedirs(f"{new_folder}/originals", exist_ok=True)
        os.makedirs(f"{new_folder}/labels", exist_ok=True)

        shutil.copy(
            f"{old_folder}/{folder}/{original_file}",
            f"{new_folder}/originals/{img_counter}.nii",
        )
        shutil.copy(
            f"{old_folder}/{folder}/{segmentation_file}",
            f"{new_folder}/labels/{img_counter}.nii",
        )


def slice_and_reformat_images(
    originals_folder: str,
    labels_folder: str,
    slicing_range: Tuple[int, int] = (50, 110),
    step: int = 10,
    new_format: str = ".png",
):
    print("Slicing and formating...")
    for image in tqdm(os.listdir(originals_folder), ncols=70):

        img = nib.load(f"{originals_folder}/{image}").get_fdata()
        label = nib.load(f"{labels_folder}/{image}").get_fdata()

        img_norm = np.zeros(img.shape)
        img_norm = cv.normalize(
            img, img_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX
        )
        label_norm = (255.0 / label.max()) * (label - label.min()).astype(int)

        for i in range(slicing_range[0], slicing_range[1] + 1, step):
            img_slice = img_norm[:, :, i]
            label_slice = label_norm[:, :, i]

            name = image.split(".")[0]
            cv.imwrite(f"{originals_folder}/{name}_{i}{new_format}", img_slice)
            cv.imwrite(f"{labels_folder}/{name}_{i}{new_format}", label_slice)

        os.remove(f"{originals_folder}/{image}")
        os.remove(f"{labels_folder}/{image}")


if __name__ == "__main__":

    OLD_FOLDER = "data/BRATS/raw/MICCAI_BraTS_2018_Data_Training/HGG"
    NEW_FOLDER = "data/BRATS"

    T1_EXTENSION = "_t1.nii"
    SEG_EXTENSION = "_seg.nii"

    move_and_rename_images(
        OLD_FOLDER,
        NEW_FOLDER,
        original_file_termination=T1_EXTENSION,
        label_file_termination=SEG_EXTENSION,
    )

    slice_and_reformat_images(f"{NEW_FOLDER}/originals", f"{NEW_FOLDER}/labels")

    resize_images(
        f"{NEW_FOLDER}/originals", f"{NEW_FOLDER}/labels", new_size=(512, 512)
    )
