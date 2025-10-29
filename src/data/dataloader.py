import numpy as np
from torch import Tensor
import os
import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .rawboost import process_rawboost_feature
from ..utils.config import get_rawboost_args
from ..utils.util import read_metadata, genSpoof_list

			
def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[ : max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[ : , : max_len][0]
    return padded_x	


class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.rawboost_args = get_rawboost_args(args)
        self.cut = 66800      

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, 'flac' , utt_id + '.flac'), sr = 16000) 
        Y = process_rawboost_feature(X, fs, self.rawboost_args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id] # 0: bonafide, 1: spoof
        return x_inp, target
    

class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key)'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800 # take ~4 sec audio 
        self.track = track

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, 'flac' , utt_id + '.flac'), sr = 16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id  
    

class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key) '''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, "wav", utt_id), sr = 16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


def get_dataloader(args, subset):
    assert subset in ['train', 'dev'], "subset must be 'train' or 'dev'"
    
    prefix      = f'ASVspoof_{args.train_track}'
    prefix_2019 = f'ASVspoof2019.{args.train_track}'

    if subset == 'train':
        shuffle = True
        drop_last = True
        sub_label, files_id_train = read_metadata(dir_meta =  os.path.join(args.protocols_path, args.train_track, f'{prefix}_cm_protocols', f'{prefix_2019}.cm.train.trn.txt'), is_eval = False)
        sub_dataset = Dataset_train(args, 
                list_IDs = files_id_train,                 
                labels = sub_label, 
                base_dir = os.path.join(args.database_path, args.train_track, '{}_{}_train/'.format(prefix_2019.split('.')[0], args.train_track)), 
                algo = args.algo)
        print('no. of train trials', len(files_id_train))

    else: # 'dev'
        shuffle = False
        drop_last = False
        sub_label, files_id_dev = read_metadata(dir_meta = os.path.join(args.protocols_path, args.train_track, f'{prefix}_cm_protocols', f'{prefix_2019}.cm.dev.trl.txt'), is_eval = False)
        sub_dataset = Dataset_train(args, 
                list_IDs = files_id_dev,
                labels = sub_label,
                base_dir = os.path.join(args.database_path, args.train_track, '{}_{}_dev/'.format(prefix_2019.split('.')[0], args.train_track)), 
                algo = args.algo)
        print('no. of validation trials', len(files_id_dev))
        
    dataloader = DataLoader(sub_dataset, batch_size = args.batch_size, num_workers = 20, shuffle = shuffle, drop_last = drop_last)
    del sub_label, sub_dataset

    return dataloader


def get_dataset(args):

    track = args.eval_track

    if track == 'In-the-Wild':
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path, track, "meta.csv"), is_train = False, is_eval = True)
        eval_set = Dataset_in_the_wild_eval(list_IDs = file_eval, base_dir = os.path.join(args.database_path, track))        
        print('no. of eval trials', len(file_eval))
    else: # 'LA' or 'DF'
        prefix      = 'ASVspoof_{}'.format(track)
        prefix_2019 = 'ASVspoof2019.{}'.format(track)
        prefix_2021 = 'ASVspoof2021.{}'.format(track)

        file_eval = read_metadata( dir_meta = os.path.join(args.protocols_path, track, f'{prefix}_cm_protocols', f'{prefix_2021}.cm.eval.trl.txt'), is_eval = True)
        eval_set = Dataset_eval(list_IDs = file_eval, base_dir = os.path.join(args.database_path, track, f'ASVspoof2021_{track}_eval/'), track = track)
        print('no. of eval trials', len(file_eval))
    
    return eval_set