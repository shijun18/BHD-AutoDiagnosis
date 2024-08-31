import os
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import shutil
import time

from trainer import VolumeClassifier
from utils import csv_reader_single,save_as_hdf5,compute_specificity
from utils import get_weight_path,get_weight_list
from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, FOLD_NUM, CSV_PATH
from config import TEST_CSV_PATH, NUM_CLASSES




def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print(f'train sample number:{len(train_id)}, val sample number:{len(validation_id)}')
    return train_id, validation_id


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train',
                        choices=["train-cross","train", "inf","inf-cross"],
                        help='choose the mode',
                        type=str)
    parser.add_argument('-s',
                        '--save',
                        default='no',
                        choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not',
                        type=str)
    parser.add_argument('-k',
                        '--key',
                        default='BHD',
                        choices=['BHD', 'CYST'],
                        help='identify target',
                        type=str)
    args = parser.parse_args()

    # Training
    ###############################################
    if 'train' in args.mode:
        assert INIT_TRAINER['is_training']
        ###### modification for new data
        csv_path = CSV_PATH
        label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')
        path_list = list(label_dict.keys())

        if args.mode == 'train-cross':
            for fold in range(1,FOLD_NUM+1):
                print('===================fold %d==================='%(fold))
                classifier = VolumeClassifier(**INIT_TRAINER)
                print(get_parameter_number(classifier.net))
                train_path, val_path = get_cross_validation(path_list, FOLD_NUM, fold)
                SETUP_TRAINER['train_path'] = train_path
                SETUP_TRAINER['val_path'] = val_path
                SETUP_TRAINER['label_dict'] = label_dict
                SETUP_TRAINER['cur_fold'] = fold

                start_time = time.time()
                classifier.trainer(**SETUP_TRAINER)

                print('run time:%.4f' % (time.time() - start_time))
        
        elif args.mode == 'train':
            classifier = VolumeClassifier(**INIT_TRAINER)
            train_path, val_path = get_cross_validation(path_list, FOLD_NUM, CURRENT_FOLD)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = label_dict
            SETUP_TRAINER['cur_fold'] = CURRENT_FOLD

            start_time = time.time()
            classifier.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))
        ###############################################

    # Inference
    ###############################################
    elif 'inf' in args.mode:
        assert not INIT_TRAINER['is_training']
        ckpt_path = SETUP_TRAINER['output_dir']

        test_csv_path = TEST_CSV_PATH
        target_names = [f'non-{args.key}',f'{args.key}']
        label_dict = csv_reader_single(test_csv_path, key_col='id', value_col='label')
        test_path = list(label_dict.keys())
        print('test len:',len(test_path))

        # set save path
        save_dir = './analysis/result/{}'.format(VERSION)
        feature_dir = './analysis/mid_feature/{}'.format(VERSION)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if args.mode == 'inf':
            INIT_TRAINER['weight_path'] = get_weight_path(
                os.path.join(ckpt_path, f'fold{str(CURRENT_FOLD)}'))
            classifier = VolumeClassifier(**INIT_TRAINER)
            print(get_parameter_number(classifier.net))

            save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}.csv')
            start_time = time.time()
            if args.save == 'no' or args.save == 'n':
                result, _, _ = classifier.inference(test_path, label_dict)
                print('run time:%.4f' % (time.time() - start_time))
            else:
                result, feature_in, feature_out = classifier.inference(
                    test_path, label_dict, hook_fn_forward=True)
                print('run time:%.4f' % (time.time() - start_time))
                # save the output of avgpool or maxpool layer
                print(feature_in.shape, feature_out.shape)
                feature_save_path = os.path.join(feature_dir,f'fold{str(CURRENT_FOLD)}')

                if os.path.exists(feature_save_path):
                    shutil.rmtree(feature_save_path)
                os.makedirs(feature_save_path)
                
                for i in range(len(test_path)):
                    name = os.path.basename(test_path[i])
                    feature_path = os.path.join(feature_save_path, name)
                    save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                    save_as_hdf5(feature_out[i], feature_path, 'feature_out')
            
            info = {}
            info['id'] = test_path
            info['true'] = result['true']
            info['pred'] = result['pred']
            for i in range(NUM_CLASSES):
                info[f'prob_{str(i+1)}'] = np.array(result['prob'])[:,i].tolist()
            csv_file = pd.DataFrame(info)
            csv_file.to_csv(save_path, index=False)
            
            #report
            print(classification_report(result['true'],result['pred'],target_names=target_names))
            cls_report = classification_report(
                result['true'],
                result['pred'],
                target_names=target_names,
                output_dict=True)
            
            specificity = compute_specificity(np.array(result['true']),np.array(result['pred']),classes=set(range(NUM_CLASSES)))
            for i,target in enumerate(target_names):
                cls_report[target]['specificity'] = specificity[i]
            cls_report['macro avg']['specificity'] = np.mean(specificity)
            
            #save as csv
            report_save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}_report.csv')
            report_csv_file = pd.DataFrame(cls_report)
            report_csv_file.to_csv(report_save_path)

            print(cls_report)
        

        elif args.mode == 'inf-cross':
            weight_path_list = get_weight_list(ckpt_path)
            for fold in range(1,FOLD_NUM+1):
                print('===================fold %d==================='%(fold))
                print('weight path: %s'%weight_path_list[fold-1])
                INIT_TRAINER['weight_path'] = weight_path_list[fold-1]
                classifier = VolumeClassifier(**INIT_TRAINER)
                print(get_parameter_number(classifier.net))
                save_path = os.path.join(save_dir,f'fold{str(fold)}.csv')
                start_time = time.time()
                if args.save == 'no' or args.save == 'n':
                    result, _, _ = classifier.inference(test_path, label_dict)
                    print('run time:%.4f' % (time.time() - start_time))
                else:
                    result, feature_in, feature_out = classifier.inference(
                        test_path, label_dict, hook_fn_forward=True)
                    print('run time:%.4f' % (time.time() - start_time))
                    # save the avgpool output
                    print(feature_in.shape, feature_out.shape)
                    feature_save_path = os.path.join(feature_dir,f'fold{str(fold)}')

                    if os.path.exists(feature_save_path):
                        shutil.rmtree(feature_save_path)
                    os.makedirs(feature_save_path)

                    for i in range(len(test_path)):
                        name = os.path.basename(test_path[i])
                        feature_path = os.path.join(feature_save_path, name)
                        save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                        save_as_hdf5(feature_out[i], feature_path, 'feature_out')
                info = {}
                info['id'] = test_path
                info['true'] = result['true']
                info['pred'] = result['pred']
                for i in range(NUM_CLASSES):
                    info[f'prob_{str(i+1)}'] = np.array(result['prob'])[:,i].tolist()
                csv_file = pd.DataFrame(info)
                csv_file.to_csv(save_path, index=False)
                
                #report
                print(classification_report(result['true'],result['pred'],target_names=target_names))
                cls_report = classification_report(
                    result['true'],
                    result['pred'],
                    target_names=target_names,
                    output_dict=True)

                specificity = compute_specificity(np.array(result['true']),np.array(result['pred']),classes=set(range(NUM_CLASSES)))
                
                for i,target in enumerate(target_names):
                    cls_report[target]['specificity'] = specificity[i]
                cls_report['macro avg']['specificity'] = np.mean(specificity)
                #save as csv
                report_save_path = os.path.join(save_dir,f'fold{str(fold)}_report.csv')
                report_csv_file = pd.DataFrame(cls_report)
                report_csv_file.to_csv(report_save_path)

                print(cls_report)

    ###############################################
