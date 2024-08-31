import shutil
import os
import pandas as pd
import h5py
import numpy as np

def compute_specificity(y_true: np.array, y_pred: np.array, classes: set = None):

    if classes is None: # Determine classes from the values
        classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))

    specs = []
    for cls in classes:
        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)

        fp = sum(y_pred_cls[y_true_cls != 1])
        tn = sum(y_pred_cls[y_true_cls == 0] == False)

        specificity_val = tn / (tn + fp)
        specs.append(specificity_val)

    return specs

def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()

def csv_reader_single(csv_file,key_col=None,value_col=None):
  '''
  Extracts the specified single column, return a single level dict.
  The value of specified column as the key of dict.

  Args:
  - csv_file: file path
  - key_col: string, specified column as key, the value of the column must be unique. 
  - value_col: string,  specified column as value
  '''
  file_csv = pd.read_csv(csv_file)
  key_list = file_csv[key_col].values.tolist()
  value_list = file_csv[value_col].values.tolist()
  
  target_dict = {}
  for key_item,value_item in zip(key_list,value_list):
    target_dict[key_item] = value_item

  return target_dict

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def remove_dir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)

def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None


def get_weight_list(ckpt_path,choice=None):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if choice is not None and eval(str(fold.name)[-1]) not in choice:
            continue
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            # print(weight_path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
    path_list.sort(key=lambda x:x.split('/')[-2])
    return path_list


def remove_weight_path(ckpt_path,retain=5):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=5):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain=retain)
        else:
            remove_weight_path(ckpt_path,retain=retain)
            break  


def print_dict_items(dict_data):
    max_key_length = max(len(key) for key in dict_data)
    for key, value in dict_data.items():
        print(f'{key.ljust(max_key_length)}: {value}')


if __name__ == '__main__':
    ckpt_path = './ckpt/'
    dfs_remove_weight(ckpt_path)
    