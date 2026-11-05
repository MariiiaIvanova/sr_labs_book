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

from tqdm import tqdm


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

# def s_norm(test_data, lines, adapt_data, N_s=200, eps=0.6):
#     """
#     Function to perform s-normalization for scores with the snorm_data
#     :param test_data: test embeddings
#     :param lines: test protocol
#     :param scores: raw scores matrix
#     :param adapt_data: data for s-norm (s-norm embeddings)
#     :param N_s: top N impostors scrores for s-normalization
#     :param eps: epsilon for std
#     :return: snorm_scores - s-normalized scores
#     """
    
#     scores_adapted = []
#     all_labels = []
#     all_trials = []

#     # Prepare lists of unique wavs from protocols
#     enroll_list = sorted(set([x.strip().split()[1] for x in lines]))
#     test_list   = sorted(set([x.strip().split()[2] for x in lines]))
#     adapt_list  = sorted(adapt_data.keys())
#     print('fGg')

#     # Prepare entolls: save enroll embds in ndarray [num_wavs x emb_size]
#     E = []
#     for id, enr in enumerate(enroll_list):
#         E.append(test_data[enr].squeeze(0).numpy())
#     E = np.array(E)

#     # Prepare tests: save test embds in ndarray [num_wavs x emb_size]
#     T = []
#     for id, tst in enumerate(test_list):
#         T.append(test_data[tst].squeeze(0).numpy())
#     T = np.array(T)

#     # Prepare adapt data: save adapt embds in ndarray [num_wavs x emb_size]
#     A = []
#     for id, a in enumerate(adapt_list):
#         A.append(adapt_data[a].squeeze(0).numpy())
#     A = np.array(A)
    
#     ###########################################################
#     # Here is your code

#     E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
#     T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-10)
#     A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)

#     # 2. Косинусные скоры эталонов и тестов с когортой
#     E_A = E @ A.T
#     T_A = T @ A.T

#     # 3. Для каждого эталона/теста: топ-N скоров с когортой → mu, sigma
#     #    np.sort по убыванию, берём первые N_s
#     def topN_stats(scores, N_s):
#         topN = np.sort(scores, axis=1)[:, -N_s:]
#         mu    = topN.mean(axis=1)
#         sigma = topN.std(axis=1, ddof=1)
#         return mu, sigma

#     mu_e, sigma_e = topN_stats(E_A, N_s)
#     mu_t, sigma_t = topN_stats(T_A, N_s)

#     # 4. Индексы для быстрого доступа: имя файла -> индекс в E/T
#     enroll_idx = {name: i for i, name in enumerate(enroll_list)}
#     test_idx   = {name: i for i, name in enumerate(test_list)}

#     # 5. Идём по протоколу
#     for line in tqdm(lines, desc='Scoring with s-norm'):
#         parts = line.strip().split()
#         if len(parts) != 3:
#             continue

#         label, enr, tst = parts[0], parts[1], parts[2]

#         i_e = enroll_idx[enr]
#         i_t = test_idx[tst]

#         # сырой скор
#         S = float(E[i_e] @ T[i_t])

#         # s-нормализация
#         S_hat = ((S - mu_e[i_e]) / (2 * (sigma_e[i_e] + eps)) +
#                  (S - mu_t[i_t]) / (2 * (sigma_t[i_t] + eps)))

#         scores_adapted.append(S_hat)
#         all_labels.append(int(label))
#         all_trials.append(f'{enr} {tst}')

#     ###########################################################

#     return scores_adapted, all_labels, all_trials

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

    # Prepare enrolls: save enroll embds in ndarray [num_wavs x emb_size]
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

    # Index maps: wav name -> row index
    enroll_idx = {name: i for i, name in enumerate(enroll_list)}
    test_idx   = {name: i for i, name in enumerate(test_list)}

    # L2-normalize all embeddings (cosine similarity = dot product)
    def l2norm(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    E = l2norm(E)
    T = l2norm(T)
    A = l2norm(A)

    # Precompute scores of each enroll/test vs the whole impostor cohort
    # Shape: [num_enroll x num_adapt], [num_test x num_adapt]
    S_E = E @ A.T   # enroll  vs adapt
    S_T = T @ A.T   # test    vs adapt

    # For each enroll/test take top-N (most difficult = highest similarity)
    # and compute mean/std
    N_eff = min(N_s, A.shape[0])

    def topN_stats(S):
        # sort along axis=1 descending
        S_sorted = np.sort(S, axis=1)[:, ::-1]
        top = S_sorted[:, :N_eff]
        mu = top.mean(axis=1)
        sigma = top.std(axis=1)
        return mu, sigma

    mu_e, sigma_e = topN_stats(S_E)
    mu_t, sigma_t = topN_stats(S_T)

    # To avoid division by zero
    sigma_e = sigma_e + eps
    sigma_t = sigma_t + eps

    # Now process each trial in the protocol
    for line in tqdm(lines, desc='Scoring with s-norm'):
        parts = line.strip().split()
        # Expect format: label enroll_wav test_wav [score]
        label  = int(parts[0])
        enr    = parts[1]
        tst    = parts[2]

        e_vec = test_data[enr].squeeze(0).numpy()
        t_vec = test_data[tst].squeeze(0).numpy()

        # cosine score between enroll and test
        e_vec = e_vec / (np.linalg.norm(e_vec) + 1e-10)
        t_vec = t_vec / (np.linalg.norm(t_vec) + 1e-10)
        S = float(np.dot(e_vec, t_vec))

        ei = enroll_idx[enr]
        ti = test_idx[tst]

        S_hat = (S - mu_e[ei]) / (2.0 * sigma_e[ei] + eps) + \
                (S - mu_t[ti]) / (2.0 * sigma_t[ti] + eps)

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
        calib_x = self.calib_params(x)
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
        self.alpha = float(np.log((1 - ptar)/ptar))

    def forward(self, target_llrs, nontarget_llrs):

        def negative_log_sigmoid(lodds):
            # Function to compute -log(sigmoid(log_odds))
            
            return torch.log1p(torch.exp(-lodds))
        
        loss_value = 0
        
        ###########################################################
        # Here is your code
        loss_value += self.ptar * negative_log_sigmoid(self.alpha - target_llrs).mean()
        loss_value += (1 - self.ptar) * negative_log_sigmoid(nontarget_llrs - self.alpha).mean()
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
            # На GPU
            tar_sc = tar_sc.cuda().float()
            imp_sc = imp_sc.cuda().float()

            # DataLoader 
            tar_sc = tar_sc.squeeze(0)
            imp_sc = imp_sc.squeeze(0)

            tar_sc = tar_sc.unsqueeze(1)
            imp_sc = imp_sc.unsqueeze(1)

            # Модель
            tar_calib = model(tar_sc).squeeze(1)
            imp_calib = model(imp_sc).squeeze(1)

            loss = criterion(tar_calib, imp_calib)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            ###########################################################
                            
        lr_value = optimizer.param_groups[0]['lr']

        if verbose:
            print("Epoch {:1.0f}, LR {:f} Loss {:f}".format(epoch, lr_value, loss.item()))
                
        scheduler[0].step()
    
    return