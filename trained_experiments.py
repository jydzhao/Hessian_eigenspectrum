import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
from IPython.display import display, Latex
from collections import defaultdict
import pandas as pd

from torch.distributions.multivariate_normal import MultivariateNormal
from datetime import datetime as dt

from networks import *
from network_derivatives import *
from utils import *
from plotting_func import *
from load_data import *

import os
from tqdm import tqdm
import yaml

from pathlib import Path
import requests
import pickle
import gzip
import scipy

import mnist_reader

def cal_hess_information(x_train, y_train, loss_func, network, calc_H, method_cond_num='naive', device='cpu', **kwargs):
     

    if calc_H:

        hessian_information = pd.DataFrame({ 
                                'input_dim':[],
                                'output_dim':[],
                                'width':[],
                                'depth':[],
                                'activ_f':[],
                                'epoch':[],
                                'H_cond':[],
                                'H_o_cond':[],
                                'lam_abs_min_H':[],
                                'lam_abs_max_H':[],
                                'lam_abs_min_H_o':[],
                                'lam_abs_max_H_o':[],
                                'mean_diff_H_H_o':[],
                                'max_diff_H_H_o':[],
                                'std_diff_H_H_o':[],
                                'H_rank':[],
                                'H_o_rank':[],
                                'H_spectrum':[],
                                'H_o_spectrum':[],
                                },dtype=object)
                                
        _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _lam_abs_min_H_o, _lam_abs_max_H_o, _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o, _H_rank, _H_o_rank, _H_spectrum, _H_o_spectrum = calc_condition_num(network,
                                                            x_train,y_train,
                                                            loss_func,
                                                            device,
                                                            calc_H,
                                                            method_cond_num)
        
#         hessian_information.append([network.input_dim, network.output_dim, 
#                                                          network.width, network.depth, network.activation_func, 
#                                                          kwargs['epoch'],
#                                                         _H_cond, _H_o_cond, 
#                                                         _lam_abs_min_H, _lam_abs_max_H, 
#                                                         _lam_abs_min_H_o, _lam_abs_max_H_o,
#                                                         _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o,
#                                                         _H_rank, _H_o_rank,
#                                                         _H_spectrum, _H_o_spectrum], ignore_index=True)
        

       
     
        hessian_information.loc[len(hessian_information)] = [network.input_dim, network.output_dim, 
                                                         network.width, network.depth, network.activation_func, 
                                                         kwargs['epoch'],
                                                        _H_cond, _H_o_cond, 
                                                        _lam_abs_min_H, _lam_abs_max_H, 
                                                        _lam_abs_min_H_o, _lam_abs_max_H_o,
                                                        _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o,
                                                        _H_rank, _H_o_rank, _H_spectrum, _H_o_spectrum]
    
    else:
       hessian_information = pd.DataFrame({ 
                                'input_dim':[],
                                'output_dim':[],
                                'width':[],
                                'depth':[],
                                'activ_f':[],
                                'epoch':[],
                                'H_o_cond':[],
                                'lam_abs_min_H_o':[],
                                'lam_abs_max_H_o':[],
                                'H_o_rank':[]
                                })
       _H_o_cond, _lam_abs_min_H_o, _lam_abs_max_H_o, _H_o_rank = calc_condition_num(network,
                                                            x_train,y_train,
                                                            loss_func,
                                                            device,
                                                            calc_H,
                                                            method_cond_num)
       hessian_information.loc[len(hessian_information)] = [network.input_dim, network.output_dim, 
                                                         network.width, network.depth, network.activation_func,
                                                        kwargs['epoch'],
                                                        _H_o_cond,
                                                        _lam_abs_min_H_o, _lam_abs_max_H_o,
                                                        _H_o_rank]
    
    
    # print('Epoch: 0 \t loss= %10.3e' %loss_func(network(x_train), y_train).detach())
      
    return hessian_information

def calc_cond_num_outer_prod_hess_explicit_1_hiddenlayer_leaky_ReLU(network, X):

    print('Starting calculation of kappa(H_O) for leaky ReLU...')
    # X: d x N
    # V.T: m x d
    # W: k x m
    k = network.output_dim
    m = network.width
    n = X.shape[1]
    print('n', n)
    print('k', k)
    
    alpha = network.activ.negative_slope
    
    print('alpha', alpha)

    V = network.lin_in.weight.detach().cpu()
    W = network.lin_out.weight.detach().cpu()
    
    XTX = X.T @ X    
    
#     print(V.device, W.device, X.device)
    
    lam_min_XTX = torch.linalg.eigvalsh(XTX)[0]
    lam_max_XTX = torch.linalg.eigvalsh(XTX)[-1]
    
    
    lam_min_WWT = torch.linalg.eigvalsh(W @ W.T)[0]
    lam_max_WWT = torch.linalg.eigvalsh(W @ W.T)[-1]

    
    lam_min_XTVVTX = torch.linalg.eigvalsh(X.T @ V.T @ V @ X)[0]
    lam_max_XTVVTX = torch.linalg.eigvalsh(X.T @ V.T @ V @ X)[-1]
    
    
    terms1 = 0
    terms2 = 0
    
    for i in range(m):
        V_i = V[i,:] # 1 x d
        V_i = torch.unsqueeze(V_i,1)
        W_i = W[:,i] # 1 x k
        W_i = torch.unsqueeze(W_i,1)
        
        tmp = V_i.T @ X

        lam_i = torch.zeros(n,n)
        for q in range(n):
            if tmp[0,q] > 0:
                lam_i[q,q] = 1
            else:
                lam_i[q,q] = -alpha
        terms2 += lam_i @ X.T @ V_i @ V_i.T @ X @ lam_i
        
        
        prod1 = lam_i @ X.T
        terms1 += torch.kron( prod1 @ prod1.T, W_i @ W_i.T)
        
     
    H_O = torch.zeros(n*k, n*k)    
    
    H_O += torch.kron(terms2, torch.eye(k)) 
    H_O += terms1
        
        
    lam_min_terms2 = torch.linalg.eigvalsh(terms2)[0]
    lam_max_terms2 = torch.linalg.eigvalsh(terms2)[-1]

#     cond_H_o_upper_bound = kappa_XTX * tr_WWT/(alpha**2 * lam_min_WWT) + sums/(alpha**2 * lam_min_XTX * lam_min_WWT)
    
    cond_H_o_upper_bound1 = (lam_max_XTX * lam_max_WWT + lam_max_terms2)/(alpha**2*lam_min_XTX * lam_min_WWT + lam_min_terms2)

    cond_H_o_upper_bound15 = (lam_max_XTX * lam_max_WWT + lam_max_XTVVTX)/(alpha**2*lam_min_XTX * lam_min_WWT + lam_min_terms2)
    
    cond_H_o_upper_bound2 = (lam_max_XTX * lam_max_WWT + lam_max_XTVVTX)/(alpha**2*lam_min_XTX * lam_min_WWT + alpha**2 * lam_min_XTVVTX)
    

    cond_H_o = torch.linalg.cond(H_O)

    

    return float(cond_H_o), float(cond_H_o_upper_bound1), float(cond_H_o_upper_bound2), float(cond_H_o_upper_bound15)

def calc_cond_num_outer_prod_hess_explicit_linearNN(d, l, k, m, network, cond_cov_xx, cov_xx):
    device='cpu'
    
    
    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on the first approximation Eq. (56/57)
    lam_max_upperb = 0
    lam_min_lowerb = 0
    
    for el in range(1,l+2+1):
        if el == l+2:
            W_1 = torch.eye(k) 
        else:
            W_1 = network.lin_out.weight.detach()
        
            for lay in range(-1,-2-l+el,-1):
                W_1 = W_1 @ network.lin_hidden[lay].weight.detach()
        
        if el == 1:
            W_2 = torch.eye(d)
        else:
            W_2 = network.lin_in.weight.detach().T
            
            for lay in range(el-2):
                W_2 = W_2 @ network.lin_hidden[lay].weight.detach().T
        
        lam_max_upperb += torch.linalg.eigvalsh(W_1 @ W_1.T)[-1] * torch.linalg.eigvalsh(W_2 @ W_2.T)[-1]
        lam_min_lowerb += torch.linalg.eigvalsh(W_1 @ W_1.T)[0] * torch.linalg.eigvalsh(W_2 @ W_2.T)[0]            
        
    cond_H_o_tilde_upper_bound_Eq5657 = ((lam_max_upperb)/(lam_min_lowerb)) * cond_cov_xx

    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on Eq. (62) 

    cond_num_term = []
    lam_min_term = []
    
    for el in range(1,l+2+1):
        if el == l+2:
            W_1 = torch.eye(k) 
        else:
            W_1 = network.lin_out.weight.detach()
        
            for lay in range(-1,-2-l+el,-1):
                W_1 = W_1 @ network.lin_hidden[lay].weight.detach()
        
        if el == 1:
            W_2 = torch.eye(d)
        else:
            W_2 = network.lin_in.weight.detach().T
            
            for lay in range(el-2):
                W_2 = W_2 @ network.lin_hidden[lay].weight.detach().T        
        
        # mat_rank_W1 = torch.linalg.matrix_rank(W_1 @ W_1.T) 
        # mat_rank_W2 = torch.linalg.matrix_rank(W_2 @ W_2.T) 
        # eigval_W1 = torch.linalg.eigvalsh(W_1 @ W_1.T) 
        # eigval_W2 = torch.linalg.eigvalsh(W_2 @ W_2.T) 
        # cond_num_term.append((eigval_W1[-1]/eigval_W1[-mat_rank_W1]) * (eigval_W2[-1]/eigval_W2[-mat_rank_W2]))

        cond_num_term.append(torch.linalg.cond(W_1 @ W_1.T) * torch.linalg.cond(W_2 @ W_2.T))
        # lam_min_term.append(torch.linalg.eigvalsh(W_1 @ W_1.T)[0] * torch.linalg.eigvalsh(W_2 @ W_2.T)[0])            
    # print(cond_num_term)
    cond_H_o_tilde_upper_bound_Eq62 = max(cond_num_term) * cond_cov_xx

    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on the second approximation Eq. (69)
    cond_H_o_tilde_upper_bound_Eq69 = torch.linalg.cond( network.lin_in.weight.detach() )**2 \
                    * torch.linalg.cond( network.lin_out.weight.detach() )**2
    
    nominator = (1/(torch.linalg.svdvals( network.lin_in.weight.detach() )[0]**2 )) \
                + (1/(torch.linalg.svdvals( network.lin_out.weight.detach() )[0]**2))
    denominator = (1/(torch.linalg.svdvals( network.lin_in.weight.detach() )[-1]**2)) \
                + (1/(torch.linalg.svdvals( network.lin_out.weight.detach() )[-1]**2))

    for lay in range(l):
        if m > max(d,k):
            cond_H_o_tilde_upper_bound_Eq69 *= (torch.linalg.svdvals( network.lin_hidden[lay].weight.detach() )[0] / \
                            torch.linalg.svdvals( network.lin_hidden[lay].weight.detach() )[max(d,k)] )**2
        else:
            cond_H_o_tilde_upper_bound_Eq69 *= (torch.linalg.svdvals( network.lin_hidden[lay].weight.detach() )[0] / \
                            torch.linalg.svdvals( network.lin_hidden[lay].weight.detach() )[-1] )**2
            
        nominator += torch.linalg.svdvals(network.lin_hidden[lay].weight.detach())[0]**2

        if m > max(d,k):
            denominator += torch.linalg.svdvals(network.lin_hidden[lay].weight.detach())[max(d,k)]**2
        else:
            denominator += torch.linalg.svdvals(network.lin_hidden[lay].weight.detach())[-1]**2
                    
    cond_H_o_tilde_upper_bound_Eq69 *= ((nominator)/(denominator)) * cond_cov_xx

    # calculate the outer Hessian product according to the expression derived by Sidak (Eq. 2)
    UT_U = torch.zeros((k*d,k*d))
    
    if l == 0:
#         V_kaiming_tmp = torch.concatenate((network.lin_in.weight.detach().to(device), torch.unsqueeze(network.lin_in.bias.detach(),axis=1)), axis=1)
#         V_kaiming = torch.zeros(V_kaiming_tmp.shape[0]+1,V_kaiming_tmp.shape[1]+1)
#         V_kaiming[:V_kaiming.shape[0]-1,:V_kaiming.shape[1]-1] = V_kaiming_tmp
#         V_kaiming[-1,-1] = 1
        
#         W_kaiming = torch.concatenate((network.lin_out.weight.detach().to(device), torch.unsqueeze(network.lin_out.bias.detach(),axis=1)), axis=1)

        V_kaiming = network.lin_in.weight.detach().to(device)
        W_kaiming = network.lin_out.weight.detach().to(device)
        
            
        H_o_tilde = torch.kron(W_kaiming@W_kaiming.T, cov_xx) + torch.kron( torch.eye(k), cov_xx**1/2 @ V_kaiming.T@V_kaiming@cov_xx**1/2)
#         H_o_tilde = torch.kron(W_kaiming@W_kaiming.T, cov_xx) + torch.kron( torch.eye(k), V_kaiming.T@V_kaiming@cov_xx)
        
    
#         H_o_test = torch.zeros((m*(k+d),m*(k+d)))
# #         H_o_test = torch.zeros((m*(k+d+2),m*(k+d+2)))
        
# #         H_o_test[0:(d+2)*(m+1),0:(d+2)*(m+1)] = torch.kron(W_kaiming.T @ W_kaiming, cov_xx)
# #         H_o_test[-k*(m+1):,-k*(m+1):] = torch.kron(torch.eye(k), V_kaiming @ cov_xx @ V_kaiming.T)
# #         H_o_test[-k*(m+1):,0:(d+2)*(m+1)] = torch.kron(W_kaiming, V_kaiming @ cov_xx)
# #         H_o_test[0:(d+2)*(m+1),-k*(m+1):] = torch.kron(W_kaiming, V_kaiming @ cov_xx).T
#         H_o_test[0:(d)*(m),0:(d)*(m)] = torch.kron(W_kaiming.T @ W_kaiming, cov_xx)
#         H_o_test[-k*(m):,-k*(m):] = torch.kron(torch.eye(k), V_kaiming @ cov_xx @ V_kaiming.T)
#         H_o_test[-k*(m):,0:(d)*(m)] = torch.kron(W_kaiming, V_kaiming @ cov_xx)
#         H_o_test[0:(d)*(m),-k*(m):] = torch.kron(W_kaiming, V_kaiming @ cov_xx).T
        
#         with open('H_O_expl.npy', 'wb') as f:
#             np.save(f, H_o_test)
            
#         eigvals = torch.sort((torch.linalg.eigvalsh(H_o_test))).values
#         print('H_o_test', eigvals)

        mat_rank = torch.linalg.matrix_rank(H_o_tilde, atol=1e-7/H_o_tilde.shape[0])
        eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
        
        print(mat_rank)
#         print(eigvals)
        if m > max(d,k):
            cond_H_o_tilde = torch.linalg.cond(H_o_tilde)
        else:
            print('use alternative')
#         print(mat_rank)
#         print(len(eigvals))
#         print(eigvals)
            cond_H_o_tilde = eigvals[-1]/eigvals[-mat_rank]
    
        with open('eigval_H_O_expl.npy', 'wb') as f:
            np.save(f, eigvals)

    else: 
        for j in range(1,l+2+1):
            if j == l+2:
                fac1 = torch.eye(k)
            else:
                fac1 = network.lin_out.weight.detach()
                
                for lay in range(-1,-2-l+j,-1):
                    fac1 = fac1 @ network.lin_hidden[lay].weight.detach()

            fac1 = fac1 @ fac1.T

            if j == 1:
                fac2 = torch.eye(d)
            else:
                fac2 = network.lin_in.weight.detach().T

                for lay in range(j-2):
                    fac2 = fac2 @ network.lin_hidden[lay].weight.detach().T

            fac2 = fac2 @ fac2.T
            
            fac1 = fac1.to(device)
            fac2 = fac2.to(device)

            UT_U += torch.kron(fac1,fac2)

        H_o_tilde = (torch.kron(torch.eye(k), cov_xx**1/2)) @ UT_U @ (torch.kron(torch.eye(k), cov_xx**1/2))

        mat_rank = torch.linalg.matrix_rank(H_o_tilde, atol=1e-7/H_o_tilde.shape[0])
        eigvals = np.sort(scipy.linalg.eigvalsh(H_o_tilde.detach().cpu()))
#         eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
#         print(mat_rank)
#         print(len(eigvals))
#         print(eigvals)
        cond_H_o_tilde = torch.linalg.cond(H_o_tilde)
#         cond_H_o_tilde = eigvals[-1]/eigvals[-mat_rank]

    return float(cond_H_o_tilde), float(cond_H_o_tilde_upper_bound_Eq5657), float(cond_H_o_tilde_upper_bound_Eq62), float(cond_H_o_tilde_upper_bound_Eq69)

def calc_cond_num_outer_prod_hess_explicit_linearResNN(d, l, k, m, network, cond_cov_xx, cov_xx):

    beta = network.beta

    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on the first approximation Eq. (56/57)
    lam_max_upperb = 0
    lam_min_lowerb = 0

    identity_in = torch.eye(network.lin_in.weight.shape[0], 
                           network.lin_in.weight.shape[1]) 
    identity_out = torch.eye(network.lin_out.weight.shape[0],
                           network.lin_out.weight.shape[1])
    if l > 0:
        identity_hidden_layer = torch.eye(network.lin_hidden[0].weight.shape[0],
                                          network.lin_hidden[0].weight.shape[1])
    
    for el in range(1,l+2+1):
        if el == l+2:
            W_1 = torch.eye(k) 
        else:
            W_1 = (network.lin_out.weight.detach() + beta *  identity_out)
        
            for lay in range(-1,-2-l+el,-1):
                W_1 = W_1 @ (network.lin_hidden[lay].weight.detach() + beta * identity_hidden_layer)
        
        if el == 1:
            W_2 = torch.eye(d)
        else:
            W_2 = (network.lin_in.weight.detach() + beta * identity_in).T
            
            for lay in range(el-2):
                W_2 = W_2 @ (network.lin_hidden[lay].weight.detach() + beta * identity_hidden_layer).T
        
        lam_max_upperb += torch.linalg.eigvalsh(W_1 @ W_1.T)[-1] * torch.linalg.eigvalsh(W_2 @ W_2.T)[-1]
        lam_min_lowerb += torch.linalg.eigvalsh(W_1 @ W_1.T)[0] * torch.linalg.eigvalsh(W_2 @ W_2.T)[0]            
        
    cond_H_o_tilde_upper_bound_Eq5657 = ((lam_max_upperb)/(lam_min_lowerb)) * cond_cov_xx

    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on Eq. (62) 

    cond_num_term = []
    lam_min_term = []
    
    for el in range(1,l+2+1):
        if el == l+2:
            W_1 = torch.eye(k) 
        else:
            W_1 = (network.lin_out.weight.detach() + beta *  identity_out)
        
            for lay in range(-1,-2-l+el,-1):
                W_1 = W_1 @ (network.lin_hidden[lay].weight.detach() + beta * identity_hidden_layer)
        
        if el == 1:
            W_2 = torch.eye(d)
        else:
            W_2 = (network.lin_in.weight.detach() + beta * identity_in).T
            
            for lay in range(el-2):
                W_2 = W_2 @ (network.lin_hidden[lay].weight.detach() + beta * identity_hidden_layer).T
        
        cond_num_term.append(torch.linalg.cond(W_1 @ W_1.T) * torch.linalg.cond(W_2 @ W_2.T))
        lam_min_term.append(torch.linalg.eigvalsh(W_1 @ W_1.T)[0] * torch.linalg.eigvalsh(W_2 @ W_2.T)[0])            
        
    # cond_num_terms.append(cond_num_term)
    cond_H_o_tilde_upper_bound_Eq62 = max(cond_num_term) * cond_cov_xx

    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on the second approximation Eq. (69)
    cond_H_o_tilde_upper_bound_Eq69 = (torch.linalg.svdvals( network.lin_in.weight.detach() )[0] + beta)**2 \
                    * (torch.linalg.svdvals( network.lin_out.weight.detach())[0] + beta)**2 / \
                    (torch.linalg.svdvals( network.lin_in.weight.detach() )[-1] + beta)**2 \
                    * (torch.linalg.svdvals( network.lin_out.weight.detach())[-1] + beta)**2
    
    nominator = (1/(torch.linalg.svdvals( network.lin_in.weight.detach() )[0] + beta)**2) \
                + (1/(torch.linalg.svdvals( network.lin_out.weight.detach() )[0]+ beta)**2)
    denominator = (1/(torch.linalg.svdvals( network.lin_in.weight.detach() )[-1]+ beta)**2) \
                + (1/(torch.linalg.svdvals( network.lin_out.weight.detach() )[-1] + beta)**2)

    for lay in range(l):
        if m > max(d,k):
            cond_H_o_tilde_upper_bound_Eq69 *= ((torch.linalg.svdvals( network.lin_hidden[lay].weight.detach() )[0] + beta) / \
                            (torch.linalg.svdvals( network.lin_hidden[lay].weight.detach() )[max(d,k)] + beta) )**2
        else:
            cond_H_o_tilde_upper_bound_Eq69 *= ((torch.linalg.svdvals( network.lin_hidden[lay].weight.detach() )[0] + beta) / \
                            (torch.linalg.svdvals(network.lin_hidden[lay].weight.detach() )[-1] + beta ) )**2
        
        nominator += (torch.linalg.svdvals(network.lin_hidden[lay].weight.detach())[0] + beta)**2

        if m > max(d,k):
            denominator += (torch.linalg.svdvals(network.lin_hidden[lay].weight.detach())[max(d,k)] + beta )**2
        else:
            denominator += (torch.linalg.svdvals(network.lin_hidden[lay].weight.detach())[-1] + beta )**2
                    
    cond_H_o_tilde_upper_bound_Eq69 *= ((nominator)/(denominator)) * cond_cov_xx

    # calculate the outer Hessian product according to the expression derived by Sidak (Eq. 2)
    UT_U = torch.zeros((k*d,k*d))
    
    if l == 0:
        V_kaiming = network.lin_in.weight.detach()
        W_kaiming = network.lin_out.weight.detach()
        
        H_o_tilde = torch.kron(( W_kaiming + beta *  identity_out) @ ( W_kaiming + beta * identity_out ).T, cov_xx) + \
                           torch.kron( torch.eye(k), ( V_kaiming + beta * identity_in ).T @  
                                                      ( V_kaiming + beta * identity_in ) @ cov_xx)

        # mat_rank = torch.linalg.matrix_rank(H_o_tilde)
        # eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
        cond_H_o_tilde = torch.linalg.cond(H_o_tilde)
        # print(eigvals)
        # cond_H_o_tilde = eigvals[-1]/eigvals[-mat_rank]
        # H_o_tilde_cond_l.append(torch.linalg.cond(H_o_tilde))
    

    else: 
        for j in range(1,l+2+1):
            if j == l+2:
                fac1 = torch.eye(k)
            else:
                fac1 = network.lin_out.weight.detach() + beta *  identity_out
                
                for lay in range(-1,-2-l+j,-1):
                    fac1 = fac1 @ (network.lin_hidden[lay].weight.detach() + beta * identity_hidden_layer)

            fac1 = fac1 @ fac1.T

            if j == 1:
                fac2 = torch.eye(d)
            else:
                fac2 = (network.lin_in.weight.detach() + beta * identity_in).T

                for lay in range(j-2):
                    fac2 = fac2 @ (network.lin_hidden[lay].weight.detach() + beta * identity_hidden_layer).T

            fac2 = fac2 @ fac2.T

            UT_U += torch.kron(fac1,fac2)

        H_o_tilde = UT_U @ (torch.kron(torch.eye(k), cov_xx))

        # mat_rank = torch.linalg.matrix_rank(H_o_tilde)
        # eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
        # print(eigvals)
        cond_H_o_tilde = torch.linalg.cond(H_o_tilde)
        # cond_H_o_tilde = eigvals[-1]/eigvals[-mat_rank]

    return float(cond_H_o_tilde), float(cond_H_o_tilde_upper_bound_Eq5657), float(cond_H_o_tilde_upper_bound_Eq62), float(cond_H_o_tilde_upper_bound_Eq69)

def eval_network_cond_num_outer_prod_hess(networks, dataset, dataset_path, device='cpu', **kwargs):

    
    
    outer_prod_hessian_information = pd.DataFrame({'dataset':[],
                                        'network':[],
                                        'cond_cov_xx':[],
                                        'input_dim':[],
                                        'output_dim':[],
                                        'width':[],
                                        'depth':[],
                                        'activ_f':[],
                                        'epoch':[],
                                        'type':[],
                                        'value':[]
                                        })
    
    if dataset == 'gaussian':
        x, _, _, _ = load_run(dataset_path)

        x = torch.tensor(x)
    elif dataset == 'mnist' or dataset == 'fashion' :
        x, _, _, _ = load_mnist(dataset, kwargs['n'], kwargs['downsample_factor'],kwargs['whiten'], device)
    elif dataset == 'cifar-10' :
        x, _, _, _ = load_cifar10(kwargs['n'], kwargs['grayscale'], kwargs['flatten'], kwargs['whiten'], device)
        
    
    n = x.shape[0]
    # calculate the empirical input covariance matrix and its condition number

#     x = torch.concatenate((x, torch.ones((n,2))),axis=1)
    cov_xx = x.T @ x / n

    cov_xx = cov_xx.to(device)

    mat_rank = torch.linalg.matrix_rank(cov_xx)
    print('n=',n)
    print('rank cov_xx: ', mat_rank)

    eigvals = torch.linalg.eigvalsh(cov_xx)
    # cond_cov_xx = float(torch.linalg.cond(cov_xx))
    cond_cov_xx = float(eigvals[-1]/eigvals[-mat_rank])
    print('cond num: ', cond_cov_xx)

    for ind, network in enumerate(networks):
        
        d = network.input_dim
        k = network.output_dim
        l = network.L
        m = network.width

        print('Network configuration: d=%d, k=%d, m=%d, L=%d' % (d,k, m, l+1))

            
        # calculate upper bounds

        if config['model_name'] == 'sequential' and network.activation_func == 'linear':
            H_o_cond, H_o_cond_bound1, H_o_cond_bound2, H_o_cond_bound3 = calc_cond_num_outer_prod_hess_explicit_linearNN(d, l, k, m, network, cond_cov_xx, cov_xx)
        elif config['model_name'] == 'lin_residual_network':
            H_o_cond, H_o_cond_bound1, H_o_cond_bound2, H_o_cond_bound3 = calc_cond_num_outer_prod_hess_explicit_linearResNN(d, l, k, m, network, cond_cov_xx, cov_xx)
        elif config['model_name'] == 'sequential' and network.activation_func == 'leaky_relu':

            H_o_cond, H_o_cond_bound1, H_o_cond_bound2, H_o_cond_bound15 = calc_cond_num_outer_prod_hess_explicit_1_hiddenlayer_leaky_ReLU(network, x.T)


        outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                    cond_cov_xx,
                                                            network.input_dim, network.output_dim, 
                                                            network.width, network.depth, network.activation_func,
                                                            kwargs['epochs'][ind],
                                                            'H_o_cond', H_o_cond
                                                            ]
        outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                    cond_cov_xx,
                                                            network.input_dim, network.output_dim,
                                                           network.width, network.depth, network.activation_func,
                                                                                   kwargs['epochs'][ind],
                                                            'H_o_cond_bound1', H_o_cond_bound1
                                                            ]
        outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                    cond_cov_xx,
                                                            network.input_dim, network.output_dim, 
                                                           network.width, network.depth, network.activation_func,
                                                                                   kwargs['epochs'][ind],
                                                            'H_o_cond_bound2', H_o_cond_bound2
                                                            ]
        
        if network.activation_func == 'leaky_relu':
            outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                        cond_cov_xx,
                                                                network.input_dim, network.output_dim, 
                                                               network.width, network.depth, network.activation_func,
                                                                                       kwargs['epochs'][ind],
                                                                'H_o_cond_bound3', H_o_cond_bound15
                                                                ]
    
    return outer_prod_hessian_information
    
def eval_network_configurations(networks, x_train, y_train, loss_func, calc_H, method_cond_num='naive', device='cpu', **kwargs):
    

    if calc_H:
        hessian_information = pd.DataFrame({'input_dim':[],
                                            'output_dim':[],
                                            'width':[],
                                            'depth':[],
                                            'activ_f':[],
                                            'epoch':[],
                                            'H_cond':[],
                                            'H_o_cond':[],
                                            'lam_abs_min_H':[],
                                            'lam_abs_max_H':[],
                                            'lam_abs_min_H_o':[],
                                            'lam_abs_max_H_o':[],
                                            'mean_diff_H_H_o':[],
                                            'max_diff_H_H_o':[],
                                            'std_diff_H_H_o':[],
                                            'H_rank':[],
                                            'H_o_rank':[]
                                            })
    else:
        hessian_information = pd.DataFrame({ 
                                'input_dim':[],
                                'output_dim':[],
                                'width':[],
                                'depth':[],
                                'activ_f':[],
                                'epoch':[],
                                'H_o_cond':[],
                                'lam_abs_min_H_o':[],
                                'lam_abs_max_H_o':[],
                                'H_o_rank':[]
                                })


#     print('Initializing Networks...')

#     print(np.linalg.cond(x_train.T@x_train))
#     print(x_train.shape)

    for ind, network in enumerate(networks):

        print('Network configuration: d=%d, k=%d, m=%d, L=%d' % (networks[0].lin_in.weight.shape[1], networks[0].lin_out.weight.shape[0], network.width, network.depth))
        
#         print(x_train.device)

        start_t = time.time()
    
        _hessian_information = cal_hess_information(x_train, y_train, 
                                                    loss_func, network, 
                                                    calc_H=calc_H,
                                                    method_cond_num=method_cond_num,
                                                    device=device, epoch=kwargs['epochs'][ind])

        hessian_information = pd.concat([hessian_information, _hessian_information],ignore_index=True)


        hessian_information['H_o_cond'] = hessian_information['H_o_cond'].astype(float)
        if calc_H:
            hessian_information['mean_diff_H_H_o'] = hessian_information['mean_diff_H_H_o'].astype(float)
            hessian_information['max_diff_H_H_o'] = hessian_information['max_diff_H_H_o'].astype(float)
            hessian_information['std_diff_H_H_o'] = hessian_information['std_diff_H_H_o'].astype(float)

        print(f'time passed: {time.time()-start_t}')
    return hessian_information

def main(project_name, experiment_name, config):

    device = config['device']
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Running experiment %s for project %s ' %(experiment_name, project_name))
    
    # Define DataFrames to log information about training process and Hessian information (condition number, eigenvalues etc.)
    dataset = config['dataset']
    if dataset == 'gaussian':
        x_train, y_train, _, _ = load_run(dataset_path)

        x_train = torch.tensor(x_train)
    elif dataset == 'mnist' or dataset == 'fashion' :
        x_train, y_train, _, _ = load_mnist(dataset, config['datapoints'], config['downsample_factor'],config['whiten'], device)
    elif dataset == 'cifar-10' :
        x_train, y_train, _, _ = load_cifar10(config['datapoints'], config['grayscale'], config['flatten'], config['whiten'], device)
    else:
        raise ValueError('Unknown dataset')
    
    outer_prod_hessian_information = pd.DataFrame({'dataset':[],
                                        'network':[],
                                        'cond_cov_xx':[],
                                        'input_dim':[],
                                        'output_dim':[],
                                        'width':[],
                                        'depth':[],
                                        'activ_f':[],
                                        'epoch':[],
                                        'type':[],
                                        'value':[]
                                        })
    
    if config['calc_H']:
        hessian_information = pd.DataFrame({'input_dim':[],
                                            'output_dim':[],
                                            'width':[],
                                            'depth':[],
                                            'activ_f':[],
                                            'epoch':[],
                                            'H_cond':[],
                                            'H_o_cond':[],
                                            'lam_abs_min_H':[],
                                            'lam_abs_max_H':[],
                                            'lam_abs_min_H_o':[],
                                            'lam_abs_max_H_o':[],
                                            'mean_diff_H_H_o':[],
                                            'max_diff_H_H_o':[],
                                            'std_diff_H_H_o':[],
                                            'H_rank':[],
                                            'H_o_rank':[],
                                            'H_spectrum':[],
                                            'H_o_spectrum':[]
                                            })
    else:
        hessian_information = pd.DataFrame({ 
                                'input_dim':[],
                                'output_dim':[],
                                'width':[],
                                'depth':[],
                                'activ_f':[],
                                'epoch':[],
                                'H_o_cond':[],
                                'lam_abs_min_H_o':[],
                                'lam_abs_max_H_o':[],
                                'H_o_rank':[]
                                })

    time_start = datetime.datetime.now()
        
    
    dataset_path = config['dataset_path']
    dataset = config['dataset']

#     epochs = config['epochs']
    epoch_max = config['epoch_max']
    every_epoch = config['every_epoch']
    if epoch_max > 1:
        epochs = np.append(np.arange(0,epoch_max,every_epoch),epoch_max-1)
    else:
        epochs = np.zeros(1)
        
    
    for filename in config['filenames']:
        Networks = [] # list of NN with different configurations
        for epoch in epochs:
            filepath = config['path']+filename + '_epoch=%d' %epoch + '.pt' 
            
            network = torch.load(filepath)
            network = network.to(device)
        # network = torch.jit.load(config['path']+filename)
        # network.eval()
            num_param = sum([len(param.flatten()) for param in network.parameters()]) 
            print(filename)
            print(f'Num parameters = {num_param}')
            
            Networks.append(network)

        if config['loss_func'] == 'mse':
            loss_func = F.mse_loss

            if config['calc_H_O_info'] == True:
                _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(Networks, dataset, dataset_path,device=device,epochs=epochs, n=config['datapoints'], downsample_factor=config['downsample_factor'], whiten=config['whiten'], flatten=config['flatten'], grayscale=config['grayscale'])
            
            if config['calc_hess_info'] == True:
                _hessian_information  = eval_network_configurations(Networks, x_train, y_train, loss_func, calc_H=config['calc_H'], method_cond_num=config['method_cond_num'],device=device, epochs=epochs)
               
        if config['calc_H_O_info'] == True:
            outer_prod_hessian_information = pd.concat([outer_prod_hessian_information, _outer_prod_hessian_information],ignore_index=True)
        
        if config['calc_hess_info'] == True:
            hessian_information = pd.concat([hessian_information, _hessian_information],ignore_index=True)


    print('Time passed: %.3f seconds' %(datetime.datetime.now()-time_start).total_seconds())         
    
    time_now = dt.now().isoformat()

    file_path = 'figures/' + config['project_name'] + '_' + config['experiment_name'] + '_' + time_now + '/'
    os.mkdir(file_path)

    if config['log_yaml_file']  == True:
        yaml_file_name = file_path + 'config_' + time_now + '.yaml'
        with open(yaml_file_name, 'w') as file:
            yaml.dump(config, file)
       
    if config['calc_H_O_info'] == True:
        outer_prod_hessian_information.to_pickle("pandas_dataframes/outer_prod_hessian_information_%s_%s.pkl" %(config['project_name'], config['experiment_name']))
        
    if config['calc_hess_info'] == True:
        hessian_information.to_pickle("pandas_dataframes/full_hessian_information_%s_%s.pkl" %(config['project_name'], config['experiment_name']))

if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)

    plt.rcParams["figure.figsize"] = (4,4)

        
    super_config = {}
    config_path = 'super_config.yaml'
    with open(config_path, 'r') as file:
        super_config.update(yaml.safe_load(file))
        
#     config_files = super_config['config_files']
    config_files = ['config_trained_experiments.yaml']
    
    for base_config_path in config_files:
        # load in YAML configuration
        config = {}

#         base_config_path = 'config_trained_experiments.yaml'
        with open(base_config_path, 'r') as file:
            config.update(yaml.safe_load(file))

        # start training with name and config 
        main(config['project_name'], config['experiment_name'], config)

