# Exercises in order to perform laboratory work

# Import of modules
import numpy as np
from math import sqrt
import itertools
import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow
from sklearn.metrics.pairwise import cosine_similarity
from common import get_eer


def get_tar_imp_scores(all_scores, all_labels):
    # Function to get target and impostors scores based on the labels

    tar_scores = []
    imp_scores = []
    for idx in range(len(all_labels)):

        if all_labels[idx] == 1:
            tar_scores.append(all_scores[idx])

        else:
            imp_scores.append(all_scores[idx])

    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)

    return tar_scores, imp_scores

def plot_histograms_2sets(all_scores_1, all_labels_1,
                          all_scores_2, all_labels_2,
                          names=['in-domain', 'out-of-domain']):
    # Function to show target/impostor histograms and compute EER to out-of-domain and in-domain datasets
    
    # Get target and impostors scores
    tar_scores_1, imp_scores_1 = get_tar_imp_scores(all_scores_1, all_labels_1)
    tar_scores_2, imp_scores_2 = get_tar_imp_scores(all_scores_2, all_labels_2)

    # Plot histograms for target and impostor scores
    min_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).min()
    max_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).max()

    hist(tar_scores_1, int(sqrt(len(tar_scores_1))), histtype='step', color='green',
         range=(min_scores, max_scores))
    hist(imp_scores_1, int(sqrt(len(imp_scores_1))), histtype='step', color='red',
         range=(min_scores, max_scores))
    hist(tar_scores_2, int(sqrt(len(tar_scores_2))), histtype='step', color='blue',
         range=(min_scores, max_scores))
    hist(imp_scores_2, int(sqrt(len(imp_scores_2))), histtype='step', color='cyan',
         range=(min_scores, max_scores))
    xlabel('$s$');
    ylabel('$\widehat{W}(s|H_0)$, $\widehat{W}(s|H_1)$');
    title('VoxCeleb1-O (cleaned), histograms');
    legend(list('{}_{}'.format(a[0], a[1]) for a in itertools.product(names, ['tar', 'imp'])))
    grid()
    show()

    # Compute equal error rate
    EER_1, thresh_EER_1 = get_eer(tar_scores_1, imp_scores_1)
    EER_2, thresh_EER_2 = get_eer(tar_scores_2, imp_scores_2)

    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f} ".format(names[0], EER_1, thresh_EER_1))
    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f} ".format(names[1], EER_2, thresh_EER_2))

def mean_embd_norm(test_embds, adapt_embds):
    # Function to apply mean embedding normalization
    
    test_embds_adapted = {}
    adapt_embds_list = [adapt_embds[k] for k in adapt_embds.keys()]
    mean_embd = torch.stack(adapt_embds_list).mean(0)
    if len(mean_embd.size()) > 1:
        mean_embd = mean_embd.mean(0)

    for k in test_embds.keys():
        test_embds_adapted[k] = test_embds[k] - mean_embd
    
    return test_embds_adapted

def s_norm(test_data, lines, adapt_data, N_s=200, eps=0.5):
    """
    Function to perform s-normalization for scores with the snorm_data
    :param test_data: test embeddings
    :param lines: test protocol
    :param scores: raw scores matrix
    :param adapt_data: data for s-norm (s-norm embeddings)
    :param N_s: top N impostors scrores for s-normalization
    :param eps: epsilon for std
    :return: snorm_scores - s-normalized scores
    """
    
    scores_adapted = []
    all_labels = []
    all_trials = []

    # Prepare lists of unique wavs from protocols
    enroll_list = list(set(list([x.strip().split()[1] for x in lines])))
    test_list   = list(set(list([x.strip().split()[2] for x in lines])))
    adapt_list  = list(adapt_data.keys())

    # Prepare entolls: save enroll embds in ndarray [num_wavs x emb_size]
    E = []
    for id, enr in enumerate(enroll_list):
        E.append(test_data[enr].squeeze(0).numpy())
    E = np.array(E)

    # Prepare tests: save test embds in ndarray [num_wavs x emb_size]
    T = []
    for id, tst in enumerate(test_list):
        T.append(test_data[tst].squeeze(0).numpy())
    T = np.array(T)

    # Prepare adapt data: save adapt embds in ndarray [num_wavs x emb_size]
    A = []
    for id, a in enumerate(adapt_list):
        A.append(adapt_data[a].squeeze(0).numpy())
    A = np.array(A)
    
    ###########################################################
    # Here is your code

    enroll_idx = {enr: i for i, enr in enumerate(enroll_list)}
    test_idx   = {tst: i for i, tst in enumerate(test_list)}
    adapt_idx  = {a: i for i, a in enumerate(adapt_list)}

    enroll_spk = {enr: enr.split('/')[0] for enr in enroll_list}
    test_spk   = {tst: tst.split('/')[0] for tst in test_list}
    adapt_spk  = {a: a.split('/')[0] for a in adapt_list}

    E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)
    T_norm = T / np.linalg.norm(T, axis=1, keepdims=True)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)

    scores_E_A = E_norm @ A_norm.T
    scores_T_A = T_norm @ A_norm.T

    # Получение mu, sigma
    def compute_cohort_stats(scores_row, spk_id, adapt_spk_list, N_s, eps):
        mask = np.array([adapt_spk_list[i] != spk_id for i in range(len(adapt_spk_list))])
        filtered = scores_row[mask]

        top_idx = np.argsort(-filtered)[:N_s]  # берем топ
        top_scores = filtered[top_idx]

        mu = top_scores.mean()
        sigma = top_scores.std() + eps #############

        return mu, sigma

    adapt_spk_list = [adapt_spk[a] for a in adapt_list]

    # mu, sigma для эталона
    enroll_stats = {}
    for enr in enroll_list:
        i = enroll_idx[enr]
        spk = enroll_spk[enr]
        mu, sigma = compute_cohort_stats(scores_E_A[i], spk, adapt_spk_list, N_s, eps)
        enroll_stats[enr] = (mu, sigma)

    # mu, sigma для теста 
    test_stats = {}
    for tst in test_list:
        j = test_idx[tst]
        spk = test_spk[tst]
        mu, sigma = compute_cohort_stats(scores_T_A[j], spk, adapt_spk_list, N_s, eps)
        test_stats[tst] = (mu, sigma)

    for line in lines:
        parts = line.strip().split()
        label = int(parts[0])
        enr = parts[1]
        tst = parts[2]
        
        e = E_norm[enroll_idx[enr]]
        t = T_norm[test_idx[tst]]
        S = float(e @ t)
        
        mu_e, sigma_e = enroll_stats[enr]
        mu_t, sigma_t = test_stats[tst]
        
        S_hat = (S - mu_e) / (2.0 * sigma_e) + (S - mu_t) / (2.0 * sigma_t)
        
        scores_adapted.append(S_hat)
        all_labels.append(label)
        all_trials.append((enr, tst))

    ###########################################################

    return scores_adapted, all_labels, all_trials

class CalibrationDataset(Dataset):

    def __init__(self, target_scores, impostor_scores):
        super(CalibrationDataset, self).__init__()

        self.target_scores   = target_scores
        self.impostor_scores = impostor_scores
        self.L_tar = len(target_scores)
        self.L_imp = len(impostor_scores)

    def __len__(self):
        
        return 1

    def __getitem__(self, idx):

        return self.target_scores, self.impostor_scores

class LinearCalibrationModel(torch.nn.Module):
    # Building of the full model for constructing the extractor of features
    
    def __init__(self):
        super(LinearCalibrationModel, self).__init__()
        
        self.calib_params = nn.Linear(1, 1)

    def forward(self, x):
        
        ###########################################################
        # Here is your code
        calib_x = self.calib_params(x)  # a·S + b, a - вес, b - смещение
        ###########################################################

        return calib_x
    
class CalibrationLoss(nn.Module):

    def __init__(self, ptar=0.01):
        '''
        Calibration loss. Code is based on https://github.com/alumae/sv_score_calibration/blob/master/calibrate_scores.py
        :param ptar: probability of target hypothesis
        '''
        
        super(CalibrationLoss, self).__init__()
        
        self.ptar  = ptar
        self.alpha = np.log(ptar/(1 - ptar))

    def forward(self, target_llrs, nontarget_llrs):

        def negative_log_sigmoid(lodds):
            # Function to compute -log(sigmoid(log_odds))
            
            return torch.log1p(torch.exp(-lodds))
        
        loss_value = 0
        
        ###########################################################
        # Here is your code
        T_act = -self.alpha  # log((1 - ptar) / ptar) - оптимальный порог

        loss_tar = self.ptar * negative_log_sigmoid(target_llrs - T_act).mean()
        loss_imp = (1 - self.ptar) * negative_log_sigmoid(T_act - nontarget_llrs).mean()

        loss_value = loss_tar + loss_imp        
        ###########################################################

        return loss_value

def train_calibration(train_loader, model, criterion, optimizer, scheduler, num_epochs, verbose=False):
    # Function to train calibration model
    
    model.train()
    
    for epoch in range(0, num_epochs):
        
        for batch_idx, batch_data in enumerate(train_loader):
            tar_sc = batch_data[0]
            imp_sc = batch_data[1]
            
            ###########################################################
            # Here is your code
            optimizer.zero_grad()

            tar_llrs = model(tar_sc.transpose(0, 1))
            imp_llrs = model(imp_sc.transpose(0, 1))

            loss = criterion(tar_llrs, imp_llrs)

            loss.backward()
            optimizer.step()
            ###########################################################
                            
        lr_value = optimizer.param_groups[0]['lr']

        if verbose:
            print("Epoch {:1.0f}, LR {:f} Loss {:f}".format(epoch, lr_value, loss.item()))
                
        scheduler[0].step()
    
    return