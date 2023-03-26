import h5py
import numpy as np
from torch.utils.data import Dataset


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image

class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - label_dict: dict, file path as key, label as value
    - transform: the data augmentation methods
    '''
    def __init__(self, path_list, label_dict, transform=None, use_roi=False):

        self.path_list = path_list
        self.label_dict = label_dict
        self.transform = transform
        self.use_roi = use_roi

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        # Get image and label
        # image: D,H,W
        # label: integer, 0,1,..
        image = hdf5_reader(self.path_list[index], 'image')
        if self.use_roi:
            mask = hdf5_reader(self.path_list[index], 'mask')
            image = self.get_roi(image,mask)
        # assert len(image.shape) == 3
        label = self.label_dict[self.path_list[index]]
        sample = {'image': image, 'label': int(label)}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    #TODO
    def get_roi(self,img,mask):
        pass