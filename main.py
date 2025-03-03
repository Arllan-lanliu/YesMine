import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import Dataset_train, Dataset_eval, Dataset_in_the_wild_eval, genSpoof_list
from model import Model
from utils import reproducibility
from utils import read_metadata
import numpy as np
import csv

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

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    correct=0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(dev_loader)
    i=0
    with torch.no_grad():
      for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        target = torch.LongTensor(batch_y).to(device)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        pred = batch_out.max(1)[1] 
        correct += pred.eq(target).sum().item()
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        i=i+1
        print("batch %i of %i (Memory: %.2f of %.2f GiB reserved) (validation)"
                  % (
                     i,
                     num_batch,
                     torch.cuda.max_memory_allocated(device) / (2 ** 30),
                     torch.cuda.max_memory_reserved(device) / (2 ** 30),
                     ),
                  end="\r",
                  )
        
    val_loss /= num_total
    test_accuracy = 100. * correct / len(dev_loader.dataset)
    print('Test accuracy: ' +str(test_accuracy)+'%')
    return val_loss

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []
    with torch.no_grad():
        for batch_x,utt_id in tqdm(data_loader,total=len(data_loader)):
            fname_list = []
            score_list = []  
            batch_x = batch_x.to(device)  
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]  
                        ).data.cpu().numpy().ravel() 
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {}\n'.format(f, cm))
            fh.close()   
        print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr,optim, device):
    num_total = 0.0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(train_loader)
    i=0
    pbar = tqdm(train_loader, total=num_batch)
    for batch_x, batch_y in pbar:    
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)     
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        i=i+1
    sys.stdout.flush()
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XLSR-Mamba')
    # Dataset
    parser.add_argument('--database_path', type=str, default='./data/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %      |- ASVspoof2021_LA_eval/wav
    %      |- ASVspoof2019_LA_train/wav
    %      |- ASVspoof2019_LA_dev/wav
    %      |- ASVspoof2021_DF_eval/wav
    '''

    parser.add_argument('--protocols_path', type=str, default='./data/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')

    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size of the model')

    parser.add_argument('--num_encoders', type=int, default=12, metavar='N',
                    help='number of encoders of the mamba blocks')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    
    # model save path
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    
    #Train
    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)

    ##===================================================Rawboost data augmentation ======================================================================#
    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. (3 for DF, 5 for LA and ITW) 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    print(args)
    args.track='LA'
 
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    track = args.track
    n_mejores=args.n_mejores_loss

    assert track in ['LA','DF','In-the-Wild'], 'Invalid track given'
    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'Bmamba{}_{}_{}_{}_ES{}_NE{}'.format(
        args.algo, track, args.loss, args.lr,args.emb_size, args.num_encoders)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    
    print('Model tag: '+ model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False

    model = model.to(device)
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
         # evaluation mode on the In-the-Wild dataset.

    if args.track == 'In-the-Wild':
        best_save_path = best_save_path.replace(track, 'LA')
        model_save_path = model_save_path.replace(track, 'LA')
        print('######## Eval ########')
        if args.average_model:
            sdl=[]
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            sd = model.state_dict()
            for i in range(1,args.n_average_model):
                model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                sd2 = model.state_dict()
                for key in sd:
                    sd[key]=(sd[key]+sd2[key])
            for key in sd:
                sd[key]=(sd[key])/args.n_average_model
            model.load_state_dict(sd)
            print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
        else:
            model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
            print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_in_the_wild_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, 'Scores/{}/{}.txt'.format(args.track, model_tag))
        sys.exit(0)
    # define train dataloader
    label_trn, files_id_train = read_metadata( dir_meta =  os.path.join(args.protocols_path+'LA/{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)), is_eval=False)
    print('no. of training trials',len(files_id_train))
    
    train_set=Dataset_train(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path+'LA/{}_{}_train/'.format(prefix_2019.split('.')[0],args.track)),algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 20, shuffle=True,drop_last = True)
    
    del train_set, label_trn
    
    # define validation dataloader
    labels_dev, files_id_dev = read_metadata( dir_meta =  os.path.join(args.protocols_path+'LA/{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix,prefix_2019)), is_eval=False)
    print('no. of validation trials',len(files_id_dev))

    dev_set = Dataset_train(args,list_IDs = files_id_dev,
		    labels = labels_dev,
		    base_dir = os.path.join(args.database_path+'LA/{}_{}_dev/'.format(prefix_2019.split('.')[0],args.track)), algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=8, num_workers=20, shuffle=False)
    del dev_set,labels_dev

    
    ##################### Training and validation #####################
    num_epochs = args.num_epochs
    not_improving=0
    epoch=0
    bests=np.ones(n_mejores,dtype=float)*float('inf')
    best_loss=float('inf')
    if args.train:
        for i in range(n_mejores):
            np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
        while not_improving<args.num_epochs:
            print('######## Epoch {} ########'.format(epoch))
            train_epoch(train_loader, model, args.lr, optimizer, device)
            val_loss = evaluate_accuracy(dev_loader, model, device)
            if val_loss<best_loss:
                best_loss=val_loss
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving=0
            else:
                not_improving+=1
            for i in range(n_mejores):
                if bests[i]>val_loss:
                    for t in range(n_mejores-1,i,-1):
                        bests[t]=bests[t-1]
                        os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                    bests[i]=val_loss
                    torch.save(model.state_dict(), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                    break
            print('\n{} - {}'.format(epoch, val_loss))
            print('n-best loss:', bests)
            epoch+=1
            if epoch>74:
                break
        print('Total epochs: ' + str(epoch) +'\n')


    print('######## Eval ########')
    if args.average_model:
        sdl=[]
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        sd = model.state_dict()
        for i in range(1,args.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key]=(sd[key]+sd2[key])
        for key in sd:
            sd[key]=(sd[key])/args.n_average_model
        model.load_state_dict(sd)
        print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
        print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

    tracks = 'LA' if args.algo == 5 else 'DF'

    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)

    if not os.path.exists('./Scores/{}'.format(tracks)):
        if not os.path.exists('./Scores'):
            os.mkdir('./Scores')
        os.mkdir('./Scores/{}'.format(tracks))

    if not os.path.exists('Scores/{}/{}.txt'.format(tracks, model_tag)):
        prefix      = 'ASVspoof_{}'.format(tracks)
        prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
        prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

        file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}/{}_cm_protocols/{}.cm.eval.trl.txt'.format(tracks, prefix,prefix_2021)), is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'{}/ASVspoof2021_{}_eval/'.format(tracks,tracks)),track=tracks)
        produce_evaluation_file(eval_set, model, device, './Scores/{}/{}.txt'.format(tracks, model_tag))
    else:
        print('Score file already exists')