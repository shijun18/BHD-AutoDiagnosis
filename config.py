from utils import print_dict_items

__all__ = ['r3d_18', 'se_r3d_18','da_18','da_se_18','r3d_34','se_r3d_34','da_34','da_se_34','vgg16_3d','vgg19_3d']

DATA_PATH = {
    'BHD': './converter/csv_file/BHD_crop_training.csv',
    'CYST': './converter/csv_file/CYST_crop_training.csv',
    'BHD_v2': './converter/csv_file/BHD_crop_training_v2.csv',
    'CYST_v2': './converter/csv_file/CYST_crop_training_v2.csv',
    'CYST_BHD_v2': './converter/csv_file/CYST_BHD_crop_training_v2.csv'
}
TEST_CSV_PATH = './converter/csv_file/BHD_crop_test.csv'

MODE = 'BHD'
CSV_PATH = DATA_PATH[MODE]
NET_NAME = 'r3d_18'
VERSION = 'v1.0'
DEVICE = '0'

IS_TRAINING = False 
USE_PRETRAINED_WEIGHT = False or 'pretrained' in VERSION


NUM_CLASSES = 2
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5


if USE_PRETRAINED_WEIGHT:
    PRETRAINED_WEIGHT_PATH = './pretrained-model/covid-19-r3d_18.pth'
else:
    PRETRAINED_WEIGHT_PATH = None

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name':NET_NAME,
    'lr':1e-3, 
    'n_epoch':100,
    'channels':1,
    'num_classes':NUM_CLASSES,
    'input_shape':(256,256,256),
    'crop':0,
    'scale':(-1600,0),
    'use_roi':False or 'roi' in VERSION,
    'batch_size':8,
    'num_workers':4,
    'device':DEVICE,
    'is_training':IS_TRAINING,
    'pretrained_weight_path':PRETRAINED_WEIGHT_PATH,
    'weight_path':None,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'gamma': 0.1,
    'milestones': [30,60,90],
    'T_max':5,
    'use_fp16':True,
    'use_maxpool': False or 'maxpool' in VERSION
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
    'output_dir':'./ckpt/{}/{}'.format(MODE,VERSION),
    'log_dir':'./log/{}/{}'.format(MODE,VERSION),
    'optimizer':'AdamW',
    'loss_fun':'Cross_Entropy',
    'class_weight':None,
    'lr_scheduler':'CosineAnnealingLR', # MultiStepLR
    'repeat_factor':3.0,
}

print_dict_items({**INIT_TRAINER, **SETUP_TRAINER})
