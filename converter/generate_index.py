import os 
import pandas as pd 
import glob
import random

INDEX = {
    'non-BHD':0,
    'BHD':1,
}

def make_label_csv(input_path,csv_path):

    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    info = []
    for subdir in os.scandir(input_path):
        # print(subdir.name)
        index = INDEX[subdir.name]
        path_list = glob.glob(os.path.join(subdir.path,"*.hdf5"))
        sub_info = [[item,index] for item in path_list]
        info.extend(sub_info)
    
    random.shuffle(info)
    # print(len(info))
    col = ['id','label']
    info_data = pd.DataFrame(columns=col,data=info)
    info_data.to_csv(csv_path,index=False)


#TODO
def merge_csv(csv_list,save_path):
    pass



if __name__ == "__main__":

    # input_path = os.path.abspath('../dataset/raw_data/resized_hdf5_file')
    input_path = os.path.abspath('../dataset/raw_data/crop_resized_hdf5_file')
    # print(input_path)
    # csv_path = './csv_file/BHD_training.csv'
    csv_path = './csv_file/BHD_crop_training.csv'
    make_label_csv(input_path,csv_path)
