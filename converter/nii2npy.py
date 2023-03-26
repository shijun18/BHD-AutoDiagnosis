import os
import h5py
import SimpleITK as sitk
import shutil
from tqdm import tqdm
import numpy as np
import copy

import nibabel as nib
from scipy.ndimage import zoom
from skimage.transform import resize
import cv2
from skimage.exposure.exposure import rescale_intensity
from skimage.draw import polygon
from skimage import measure
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, label

def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()

def nii_reader(data_path):
    data = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(data).astype(np.float32)
    return image


def get_resampled_nii(nii_file, new_spacing=[1,1,1]):
    """
    Resample NIfTI image to new spacing.
    :param nii_file: path to NIfTI file
    :param new_spacing: new voxel spacing
    :return: resampled NIfTI image
    """
    # Load NIfTI image
    nii = nib.load(nii_file)
    data = nii.get_fdata()
    affine = nii.affine
    # Calculate current voxel spacing
    current_spacing = np.abs(np.diag(affine)[:3])
    # print(current_spacing)
    # Calculate zoom factor
    zoom_factor = current_spacing / new_spacing
    # print(zoom_factor)
    # Resample image
    resampled_data = zoom(data, zoom_factor, order=1)
    # print(resampled_data.shape)
    # Update affine matrix
    new_affine = np.copy(affine)
    new_affine[:3, :3] = np.diag(new_spacing)
    # print(new_affine)
    # Save resampled NIfTI image
    resampled_nii = nib.Nifti1Image(resampled_data, new_affine, header=nii.header)
    return resampled_nii


def crop_data(image):
    print(image.shape)
    # Threshold image, threshold = 0 (water)
    # threshold = np.percentile(image, 99)
    # binary = image > threshold
    binary = image > 0
    # # Fill holes in binary image
    filled = binary_fill_holes(binary)

    # # Erode and dilate binary image
    eroded = binary_erosion(filled)
    dilated = binary_dilation(eroded)

    # Find largest connected component
    labeled, num_labels = label(dilated)
    largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    largest_component = labeled == largest_label

    # Crop image to bounding box of largest connected component
    x, y, z = np.where(largest_component)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()
    # print(ymin, ymax,zmin, zmax)
    cropped_image = image[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    print(cropped_image.shape)
    return cropped_image    



def resample_nii(input_dir,save_dir,new_spacing=[1,1,1]):

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    sample_path = [case.path for case in os.scandir(input_dir)]

    for sample in tqdm(sample_path):
        save_path = os.path.join(save_dir,os.path.basename(sample))
        resampled_nii = get_resampled_nii(sample, new_spacing)
        nib.save(resampled_nii,save_path)



def nii2npy(input_dir,save_dir,do_resize=False,do_crop=False,target_size=(256,256,256)):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    sample_path = [case.path for case in os.scandir(input_dir)]

    for sample in tqdm(sample_path):
        sample_id = os.path.basename(sample).split('.')[0]
        save_path = os.path.join(save_dir,f'{sample_id}.hdf5')
        images = nii_reader(sample)
        # print(images.shape)
        # print(np.max(images),np.min(images))
        if do_crop:
            images = crop_data(images)
            do_resize = True
        if do_resize:
            images = resize(images, target_size, mode='constant',anti_aliasing=True)
            # images = zoom(images, zoom=np.array(target_size) / np.array(images.shape))
        save_as_hdf5(images,save_path,'image')
        # break


if __name__ == "__main__":

    ## step1:  resample the data to standard pixel spacing of [1.,1.,1.]
    # input_dir = '../dataset/raw_data/nii_file/BHD'
    # save_dir = '../dataset/raw_data/resampled_nii_file/BHD'
    # resample_nii(input_dir,save_dir,[1.,1.,1.])

    # input_dir = '../dataset/raw_data/nii_file/non-BHD'
    # save_dir = '../dataset/raw_data/resampled_nii_file/non-BHD'
    # resample_nii(input_dir,save_dir,[1.,1.,1.])

    ## step2:  convert nii data to hdf5 format
    # input_dir = '../dataset/raw_data/resampled_nii_file/BHD'
    # save_dir = '../dataset/raw_data/resized_hdf5_file/BHD'
    # nii2npy(input_dir,save_dir,do_resize=True,target_size=(256,256,256))

    # input_dir = '../dataset/raw_data/resampled_nii_file/non-BHD'
    # save_dir = '../dataset/raw_data/resized_hdf5_file/non-BHD'
    # nii2npy(input_dir,save_dir,do_resize=True,target_size=(256,256,256))

    input_dir = '../dataset/raw_data/resampled_nii_file/BHD'
    save_dir = '../dataset/raw_data/crop_resized_hdf5_file/BHD'
    nii2npy(input_dir,save_dir,do_resize=True,do_crop=True,target_size=(256,256,256))

    input_dir = '../dataset/raw_data/resampled_nii_file/non-BHD'
    save_dir = '../dataset/raw_data/crop_resized_hdf5_file/non-BHD'
    nii2npy(input_dir,save_dir,do_resize=True,do_crop=True,target_size=(256,256,256))