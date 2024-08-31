import os
import h5py
import numpy as np
import pandas as pd

def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image

max_list = []
min_list = []
file_name = []

file_path = '../dataset/raw_data/resized_hdf5_file/non-BHD'
crop_file_path = '../dataset/raw_data/crop_resized_hdf5_file/non-BHD'
for sample in os.scandir(file_path):
    file_name.append(sample.name)
    image = hdf5_reader(sample.path,'image')
    max_list.append(np.max(image))
    min_list.append(np.min(image))
    # crop_image = hdf5_reader(os.path.join(crop_file_path,sample.name),'image')
    # print(np.max(crop_image),np.min(crop_image))


file_path = '../dataset/raw_data/resized_hdf5_file/BHD'
crop_file_path = '../dataset/raw_data/crop_resized_hdf5_file/BHD'
for sample in os.scandir(file_path):
    file_name.append(sample.name)
    image = hdf5_reader(sample.path,'image')
    max_list.append(np.max(image))
    min_list.append(np.min(image))
    # print(np.max(image),np.min(image))
    # crop_image = hdf5_reader(os.path.join(crop_file_path,sample.name),'image')
    # print(np.max(crop_image),np.min(crop_image))

df = pd.DataFrame(data = {'file':file_name,'max':max_list,'min':min_list})
df.to_csv('./max-min.csv',index=False)