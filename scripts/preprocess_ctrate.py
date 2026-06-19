import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
import csv
from multiprocessing import Pool, cpu_count


def read_nii_files(directory):
    nii_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files


def load_and_orient_to_RAS(file_path: str) -> np.ndarray:
    img = nib.load(file_path)
    spacing = img.header.get_zooms()  # (x, y, z)

    data = img.get_fdata()
    original = data.shape
    affine = img.affine

    current_ornt = io_orientation(affine)
    target_ornt = axcodes2ornt(('R', 'A', 'S'))  # target = Right-Anterior-Superior

    transform = ornt_transform(current_ornt, target_ornt)
    data_ras = nib.orientations.apply_orientation(data, transform)

    data_zyx = np.transpose(data_ras, (2, 1, 0))  # da X,Y,Z → Z,Y,X

    return data_zyx.astype(np.float32), spacing


def clip_and_resample(img_data, original_spacing, target_spacing=(3.0, 0.75, 0.75)):
    img_data = np.clip(img_data, -1000, 1000).astype(np.float32)

    resize_factors = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(np.array(img_data.shape) * resize_factors).astype(int)

    # Resize in PyTorch

    img_tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    resized_tensor = F.interpolate(
        img_tensor, size=tuple(new_shape), mode='trilinear', align_corners=False
    )

    final_tensor = F.interpolate(resized_tensor, size=(128, 512, 512), mode='trilinear', align_corners=False)

    resized = final_tensor.squeeze().numpy()

    return resized.astype(np.int16)


def save_nifti(array, save_path, spacing=(3.0, 0.75, 0.75)):
    # Rearrange to (X, Y, Z) for nibabel (axis -1 = Z)
    array = np.transpose(array, (2, 1, 0))  # (Z, Y, X) → (X, Y, Z)

    # Define affine using spacing
    affine = np.diag([*spacing[::-1], 1.0])  # invert because nibabel uses X,Y,Z

    img_nifti = nib.Nifti1Image(array, affine=affine)
    nib.save(img_nifti, save_path)


def process_file(file_path):
    split = 'train' # or valid
    try:
        save_path = file_path.replace(f'/{split}_fixed/', f'/{split}_fixed_preprocessed/')
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)

        img_data, spacing = load_and_orient_to_RAS(file_path)
        img_processed = clip_and_resample(img_data, original_spacing=spacing[::-1])
        save_nifti(img_processed, save_path)
        os.remove(file_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        with open("preprocess_error.csv", "a") as f:
            f.write(f"{file_path}\n")


if __name__ == "__main__":

    input_dir = "./dataset/train_fixed"
    nii_files = sorted(read_nii_files(input_dir))
    print(len(nii_files))

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, nii_files), total=len(nii_files)))
