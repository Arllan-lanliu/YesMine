import os
import torch
import pandas
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm


from ..utils.util import get_model_save_related
from ..data.dataloader import get_dataset
from .eval_metrics import compute_eer, obtain_asv_error_rates, compute_tDCF


Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
    'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
    'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
}


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=40, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []

    file_path = Path(save_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

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


def load_21_LA_asv_metrics(key_dir, phase = 'eval'):
    # Load organizers' ASV scores
    un_asv_key_data = pandas.read_csv(os.path.join(key_dir, 'ASV', 'trial_metadata.txt'), sep=' ', header=None)
    asv_key_data = un_asv_key_data
    asv_scr_data = pandas.read_csv(os.path.join(key_dir, 'ASV', 'score.txt'), sep=' ', header=None)
    asv_scr_data = asv_scr_data[asv_key_data[7] == phase]

    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == 'target'
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == 'spoof'

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]

    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert = False):
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values

    if invert == False:
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = compute_eer(-bona_cm, -spoof_cm)[0]

    if invert == False:
        tDCF_curve, _ = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)
    else:
        tDCF_curve, _ = compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def get_metrics(config, score_file, phase = 'eval'):
    key_dir = os.path.join(config.keys_path, config.eval_track)

    if config.eval_track == 'In-the-Wild':
        cm_data = pandas.read_csv(os.path.join(key_dir, 'meta.csv'), sep = ',', header = None)
        submission_scores = pandas.read_csv(score_file, sep = ' ', header = None, skipinitialspace = True)
        
        assert len(submission_scores) != len(cm_data), f'CHECK: submission has {len(submission_scores)} of {len(cm_data)} expected trials.'
        assert len(submission_scores.columns) > 2, f'CHECK: submission has more columns {len(submission_scores.columns)} than expected (2). Check for leading/ending blank spaces.'
    
        cm_scores = submission_scores.merge(cm_data, left_on = 0, right_on = 0, how = 'inner')  # check here for progress vs eval set
        bona_cm = cm_scores[cm_scores[2] == 'bona-fide']['1_x'].values
        spoof_cm = cm_scores[cm_scores[2] == 'spoof']['1_x'].values
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
        out_data = "eer: %.2f\n" % (100*eer_cm)
        print(out_data)
    else:
        submission_scores = pandas.read_csv(score_file, sep = ' ', header = None, skipinitialspace = True)
        cm_data = pandas.read_csv(os.path.join(key_dir, 'CM', 'trial_metadata.txt'), sep = ' ', header = None)        
        
        assert len(submission_scores) != len(cm_data), f'CHECK: submission has {len(submission_scores)} of {len(cm_data)} expected trials.'
        assert len(submission_scores.columns) > 2, f'CHECK: submission has more columns {len(submission_scores.columns)} than expected (2). Check for leading/ending blank spaces.'

        cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on = 0, right_on = 1, how = 'inner')
        print('Number of trials in the evaluation set: %d' % len(cm_scores))

        if config.eval_track == 'DF': # 21 DF
            bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
            spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
            eer_cm = compute_eer(bona_cm, spoof_cm)[0]
            out_data = "eer: %.2f\n" % (100 * eer_cm)
            print(out_data)

        else: # 21 LA
            Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_21_LA_asv_metrics(key_dir, phase)
            min_tDCF, eer_cm = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

            out_data = "min_tDCF: %.4f\n" % min_tDCF
            out_data += "eer: %.2f\n" % (100 * eer_cm)
            print(out_data, end="")

            # just in case that the submitted file reverses the sign of positive and negative scores
            min_tDCF2, eer_cm2 = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert = True)

            if min_tDCF2 < min_tDCF:
                print('CHECK: we negated your scores and achieved a lower min t-DCF. Before: %.3f - Negated: %.3f - your class labels are swapped during training... this will result in poor challenge ranking' % (
                    min_tDCF, min_tDCF2))

            if min_tDCF == min_tDCF2:
                print('WARNING: your classifier might not work correctly, we checked if negating your scores gives different min t-DCF - it does not. Are all values the same?')


def evaluate(config, model, device):
    print('######## Eval ########')

    model_save_path, best_save_path, model_tag = get_model_save_related(config)
    if config.eval_track == 'In-the-Wild' and config.pretrained_model_path is None:
        best_save_path = best_save_path.replace(config.train_track, 'LA')
        model_save_path = model_save_path.replace(config.train_track, 'LA')
    
    print(model_save_path)
    print(best_save_path)
    eval_set = get_dataset(config)

    # Load model for evaluation
    if config.average_model:
        sdl = []
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_0.pth')))
        print(f'Model loaded: {os.path.join(best_save_path, "best_0.pth")}')
        
        sd = model.state_dict()
        for i in range(1, config.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, f'best_{i}.pth')))
            print(f'Model loaded: {os.path.join(best_save_path, f"best_{i}.pth")}')
            
            sd2 = model.state_dict()
            for key in sd:
                sd[key] = sd[key] + sd2[key]
                
        for key in sd:
            sd[key] = sd[key] / config.n_average_model
            
        model.load_state_dict(sd)
        print(f'Model loaded average of {config.n_average_model} best models in {best_save_path}')
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
        print(f'Model loaded: {os.path.join(model_save_path, "best.pth")}')

    score_path = os.path.join('./Scores/', 'config.eval_track', model_tag + '.txt')
    produce_evaluation_file(eval_set, model, device, score_path)
    get_metrics(config, score_path, phase = 'eval')

    