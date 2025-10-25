import numpy as np
import os
import torch
import random
import csv
import torch
import numpy as np


def read_metadata(dir_meta, is_eval=False):
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
    
    if (is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list


def load_metadata(csv_path):
    label_dict = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file,delimiter=' ')
        for row in reader:
            # filename, _, label = row  # Ignore the middle column (speaker name)
            filename = row[1]
            label = row[5]
            label_dict[filename] = label
    return label_dict


def reproducibility(random_seed, args=None):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to False")
    print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()[1:] 

    if (is_train):
        for line in l_meta:
             key,_,label = line.strip().split(",")
             file_list.append(key)
             d_meta[key] = 1 if label == 'bona-fide' else 0
        return d_meta,file_list
    elif(is_eval):
        for line in l_meta:
            xline = line.strip()
            key,_,label = line.strip().split(",")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             key,_,label = line.strip().split(",")
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bona-fide' else 0
        return d_meta,file_list


def get_model_save_related(config):
    #define model saving path
    if config.model_tag:
        model_tag = config.model_tag
    else:
        model_tag = 'Bmamba{}_{}_{}_{}_ES{}_NE{}'.format(config.algo, config.train_track, config.loss, config.lr,config.emb_size, config.num_encoders)
        if config.comment:
            model_tag = model_tag + '_{}'.format(config.comment)
        if config.comment_eval:
            model_tag = model_tag + '_{}'.format(config.comment_eval)

    if config.pretrained_model_path:
        model_save_path = config.pretrained_model_path
    else:
        model_save_path = os.path.join('models', model_tag)
    best_save_path = os.path.join(model_save_path, 'best')

    return model_save_path, best_save_path, model_tag


# def my_collate(batch): #Dataset return sample = (utterance, target, nameFile) #shape of utterance [1, lenAudio]
#   data = [dp[0] for dp in batch]
#   label = [dp[1] for dp in batch]
#   nameFile = [dp[2] for dp in batch]
#   return (data, label, nameFile) 