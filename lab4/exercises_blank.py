# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow, xlim, ylim, gca


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    # Here is your code
    for score, label in zip(all_scores, all_labels):
        if label == 1:
            tar_scores.append(score)   # таргет
        else:
            imp_scores.append(score)   # импостор 
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    idx = np.argsort(all_scores)

    all_scores_sort   = all_scores[idx]
    ground_truth_sort = all_labels[idx].astype(bool)
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)
    LLR = np.log(tar_gauss_pdf / (imp_gauss_pdf + 1e-10))
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlim(-20, 30)
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    len_thr = len(LLR)

    ###########################################################
    # Here is your code
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
        
    for idx in range(len_thr):
        solution = LLR > LLR[idx]   # Порог среди значений LLR
        err = (solution != ground_truth_sort)
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # FNR
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # FPR
    
    idx = np.argmin(np.abs(fnr_thr - fnr))
    
    thr = LLR[idx]
    fpr = fpr_thr[idx]
    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0  # средняя стоимость (байесовский риск)
    
    ###########################################################
    # Here is your code
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    AC_thr  = np.zeros(len_thr)

    for idx in range(len_thr):
        solution =  LLR > LLR[idx]
        err = (solution != ground_truth_sort)
        
        fnr_thr[idx] = np.sum(err[ground_truth_sort])/len(tar_scores) # FNR
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # FPR
        
        AC_thr[idx] = (C00 * (1 - fnr_thr[idx]) * P_Htar + C10 * fnr_thr[idx] * P_Htar + C01 * fpr_thr[idx] * (1 - P_Htar) + C11 * (1 - fpr_thr[idx]) * (1 - P_Htar))

    idx = np.argmin(AC_thr)
    
    thr = LLR[idx]
    fnr = fnr_thr[idx]
    fpr = fpr_thr[idx]
    AC  = AC_thr[idx]

    plot(LLR, AC_thr, color='blue')
    xlim(-20, 30)
    xlabel('$LLR$'); ylabel('$\\overline{C}$'); title('Average cost'); grid(); show()

    ###########################################################
    
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0

    ###########################################################
    # Here is your code    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    AC_matrix = np.zeros((len_thr, len(P_Htar_thr)))

    for idx in range(len_thr):

        solution = LLR > LLR[idx]
        err = solution != ground_truth_sort

        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / len(tar_scores)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)

        for p_idx, p in enumerate(P_Htar_thr):
            AC_matrix[idx, p_idx] = (C00 * (1 - fnr_thr[idx]) * p + C10 * fnr_thr[idx] * p + C01 * fpr_thr[idx] * (1 - p) + C11 * (1 - fpr_thr[idx]) * (1 - p))

    # Максимальный риск для каждого порога
    max_AC_thr = np.max(AC_matrix, axis=1)

    # Минимум максимального риска
    idx = np.argmin(max_AC_thr)

    thr = LLR[idx]
    fnr = fnr_thr[idx]
    fpr = fpr_thr[idx]
    AC = max_AC_thr[idx]

    max_p_idx = np.argmax(AC_matrix[idx, :])
    P_Htar = P_Htar_thr[max_p_idx] 

    imshow (AC_matrix.T, aspect='auto', origin='lower', extent=[P_Htar_thr.min(), P_Htar_thr.max(), LLR.min(), LLR.max()])
    ylabel('$LLR$')
    xlabel('$P(H_{0})$')
    title('Average cost')
    show()

    ###########################################################

    return thr, fnr, fpr, AC, P_Htar