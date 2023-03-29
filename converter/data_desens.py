import os 
import shutil
import pandas as pd
import SimpleITK as sitk
import pydicom
import nibabel as nib
from tqdm import tqdm
import numpy as np

def modify_file_name(input_dir):
    
    modify_dict = {
        'before':[],
        'after':[]
    }
    for subdir in os.scandir(input_dir):
    # if subdir.name == 'CT3':
        item_list = [case.path for case in os.scandir(subdir.path)]
        # print(item_list)
        for index, item in tqdm(enumerate(item_list)):
            new_name = os.path.join(subdir.path, f'{subdir.name}_{index}')
            modify_dict['before'].append(item)
            modify_dict['after'].append(new_name)
            # print(item,new_name)
            os.rename(item,new_name)
    df = pd.DataFrame(data=modify_dict)
    print(df)
    df.to_csv('./modify.csv',mode='a',index=False)




def dcm_to_nii(input_dir,save_dir):
    '''subdir structure like:
    --CT0
      --CT0_1
        --*.dcm 
    --CTn
    '''
    def convert_to_nii(directory_path,output_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, output_path)

    # def convert_to_nii(dcm_dir, nii_file):
    #     # Get list of DICOM files
    #     dcm_files = [os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir)]
    #     meta_data = [pydicom.read_file(dcm,force=True) for dcm in dcm_files]
    #     for i in range(len(meta_data)):
    #         meta_data[i].file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    #     meta_data.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    #     images = np.stack([s.pixel_array for s in meta_data],axis=0).astype(np.float32)
    #     # print(images.shape)
    #     nifti_data = nib.Nifti1Image(images, np.eye(4))
    #     # # Save NIfTI file
    #     nib.save(nifti_data, nii_file)


    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    dcm_path = []
    nii_path = []
    for subdir in os.scandir(input_dir):
        if subdir.name == 'CT3':
            for item in os.scandir(subdir.path):
                dcm_path.append(item.path)
                nii_path.append(os.path.join(save_dir,f'{item.name}.nii.gz'))
    error_data =[]
    for dcm_path, nii_path in tqdm(zip(dcm_path,nii_path)):
        # print(dcm_path,nii_path)
        try:
            convert_to_nii(dcm_path,nii_path)
        except:
            error_data.append(dcm_path)
        
    print(error_data)


def get_data_attr(input_dir,save_csv):
    def read_nii_file(file_path):
        # load the .nii file
        img = sitk.ReadImage(file_path)
        # get the data and attributes
        size = list(img.GetSize()[:2])
        z_size = img.GetSize()[-1]
        thickness = img.GetSpacing()[-1]
        pixel_spacing = list(img.GetSpacing()[:2])
        # return the attributes
        return [size, z_size, thickness, pixel_spacing]
    info = []
    sample_list = [case.path for case in os.scandir(input_dir)]
    for item in tqdm(sample_list):
        info_item = [os.path.basename(item)]
        info_item.extend(read_nii_file(item))
        info.append(info_item)
    col = ['filename', 'size', 'slices', 'thickness', 'pixel_spacing']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(save_csv, mode='a', index=False)
    



if __name__ == "__main__":

    ## step1: modify the dir name to anonymize
    # input_dir = '../dataset/raw_data/dcm_file/BHD'
    # input_dir = '../dataset/raw_data/dcm_file/non-BHD'
    # modify_file_name(input_dir)

    # input_dir = '../dataset/raw_data/dcm_file/BHD'
    # modify_file_name(input_dir)

    ## step 2: convert dcm series to nii for data desensitization
    input_dir = '../dataset/raw_data/dcm_file/BHD'
    save_dir = '../dataset/raw_data/nii_file/BHD-miss'
    # input_dir = '../dataset/raw_data/dcm_file/non-BHD'
    # save_dir = '../dataset/raw_data/nii_file/non-BHD'

    dcm_to_nii(input_dir,save_dir)

    ## step 3: get the attrs of nii data
    # input_dir = '../dataset/raw_data/nii_file/BHD'
    # input_dir = '../dataset/raw_data/nii_file/non-BHD'
    # save_csv = './data_attr.csv'

    # get_data_attr(input_dir,save_csv)