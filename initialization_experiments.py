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

from datetime import datetime as dt

from networks import *
from network_derivatives import *
from utils import *
from plotting_func import *
from load_data import *

import os
from tqdm import tqdm
import yaml

import pickle

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

    V = network.lin_in.weight.detach()
    W = network.lin_out.weight.detach()
    
    XTX = X.T @ X    
    
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

def calc_cond_num_outer_prod_hess_explicit_linearNN(d, l, k, m, network, cond_cov_xx, cov_xx, device):

    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on the first approximation 
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
    # based on second approximation

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
        
        cond_num_term.append(torch.linalg.cond(W_1 @ W_1.T) * torch.linalg.cond(W_2 @ W_2.T))
        lam_min_term.append(torch.linalg.eigvalsh(W_1 @ W_1.T)[0] * torch.linalg.eigvalsh(W_2 @ W_2.T)[0])            
        
    # cond_num_terms.append(cond_num_term)
    cond_H_o_tilde_upper_bound_Eq62 = max(cond_num_term) * cond_cov_xx

    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on the third approximation
    
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
    UT_U = torch.zeros((k*d,k*d)).to(device)
    
    I_k = torch.eye(k).to(device)
    I_d = torch.eye(d).to(device)
    
    if l == 0:
        V_kaiming = network.lin_in.weight.detach()
        W_kaiming = network.lin_out.weight.detach()
        
#         print(cov_xx.device)

  
        H_o_tilde = torch.kron(W_kaiming@W_kaiming.T, cov_xx) + torch.kron( I_k, V_kaiming.T@V_kaiming@cov_xx)

        mat_rank = torch.linalg.matrix_rank(H_o_tilde)
        eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
        # cond_H_o_tilde = torch.linalg.cond(H_o_tilde)
        # print(eigvals)
#         print('cond=',eigvals[-1]/eigvals[-mat_rank])
        cond_H_o_tilde = eigvals[-1]/eigvals[-mat_rank]
#         H_o_tilde_cond_l.append(torch.linalg.cond(H_o_tilde))
    

    else: 
        for j in range(1,l+2+1):
            if j == l+2:
                fac1 = I_k
            else:
                fac1 = network.lin_out.weight.detach()
                
                for lay in range(-1,-2-l+j,-1):
                    fac1 = fac1 @ network.lin_hidden[lay].weight.detach()

            fac1 = fac1 @ fac1.T

            if j == 1:
                fac2 = I_d
            else:
                fac2 = network.lin_in.weight.detach().T

                for lay in range(j-2):
                    fac2 = fac2 @ network.lin_hidden[lay].weight.detach().T

            fac2 = fac2 @ fac2.T

            UT_U += torch.kron(fac1,fac2)

        H_o_tilde = UT_U @ (torch.kron(I_k, cov_xx))

        mat_rank = torch.linalg.matrix_rank(H_o_tilde)
        eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
#         print(mat_rank)
#         print(eigvals[-1]/eigvals[-mat_rank])
        # print(eigvals)
        # cond_H_o_tilde = torch.linalg.cond(H_o_tilde)
        cond_H_o_tilde = eigvals[-1]/eigvals[-mat_rank]

    return float(cond_H_o_tilde), float(cond_H_o_tilde_upper_bound_Eq5657), float(cond_H_o_tilde_upper_bound_Eq62), float(cond_H_o_tilde_upper_bound_Eq69)

def calc_cond_num_outer_prod_hess_explicit_linearResNN(d, l, k, m, network, cond_cov_xx, cov_xx, device):

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

        mat_rank = torch.linalg.matrix_rank(H_o_tilde)
        eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
        cond_H_o_tilde = torch.linalg.cond(H_o_tilde)
        # print(eigvals)
#         cond_H_o_tilde = eigvals[-1]/eigvals[-mat_rank]
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

        mat_rank = torch.linalg.matrix_rank(H_o_tilde)
        eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
        # print(eigvals)
        cond_H_o_tilde = torch.linalg.cond(H_o_tilde)

    return float(cond_H_o_tilde), float(cond_H_o_tilde_upper_bound_Eq5657) , float(cond_H_o_tilde_upper_bound_Eq62), float(cond_H_o_tilde_upper_bound_Eq69)


def cal_hess_information(x_train, y_train, loss_func, network, calc_H, method_cond_num='naive', device='cpu'):
     

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
                                'H_o_rank':[]
                                })
                                
        _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _lam_abs_min_H_o, _lam_abs_max_H_o, _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o, _H_rank, _H_o_rank = calc_condition_num(network,
                                                            x_train,y_train,
                                                            loss_func,
                                                            device,
                                                            calc_H,
                                                            method_cond_num)
        hessian_information.loc[len(hessian_information)] = [network.input_dim, network.output_dim, 
                                                         network.width, network.depth, network.activation_func, 0,
                                                        _H_cond, _H_o_cond, 
                                                        _lam_abs_min_H, _lam_abs_max_H, 
                                                        _lam_abs_min_H_o, _lam_abs_max_H_o,
                                                        _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o,
                                                        _H_rank, _H_o_rank]
    
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
                                                         network.width, network.depth, network.activation_func, 0,
                                                        _H_o_cond,
                                                        _lam_abs_min_H_o, _lam_abs_max_H_o,
                                                        _H_o_rank]
    
    
    # print('Epoch: 0 \t loss= %10.3e' %loss_func(network(x_train), y_train).detach())
      
    return hessian_information
   

def eval_network_configurations(num_inits, init_type, networks, x_train, y_train, loss_func, m_L_config, calc_H, method_cond_num='naive', seed=314159, device='cpu'):
    
    # Define DataFrames to log information about training process and Hessian information (condition number, eigenvalues etc.)

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


    print('Initializing Networks...')

    for ind, network in enumerate(networks):

        print('Network configuration: d=%d, k=%d, m=%d, L=%d' % (networks[0].lin_in.weight.shape[1], networks[0].lin_out.weight.shape[0], m_L_config[ind][0], m_L_config[ind][1]+1))

        torch.manual_seed(seed)

        for _ in range(num_inits):

            network.init_weights(init_type)
            
        
            _hessian_information = cal_hess_information(x_train, y_train, 
                                                        loss_func, network, 
                                                        calc_H=calc_H,
                                                        method_cond_num=method_cond_num,
                                                        device=device)
                
            hessian_information = pd.concat([hessian_information, _hessian_information],ignore_index=True)

        
            hessian_information['H_o_cond'] = hessian_information['H_o_cond'].astype(float)
            if calc_H:
                hessian_information['mean_diff_H_H_o'] = hessian_information['mean_diff_H_H_o'].astype(float)
                hessian_information['max_diff_H_H_o'] = hessian_information['max_diff_H_H_o'].astype(float)
                hessian_information['std_diff_H_H_o'] = hessian_information['std_diff_H_H_o'].astype(float)

    return hessian_information

def eval_network_cond_num_outer_prod_hess(num_inits, init_type, networks, dataset, seed=314159, device='cpu', **kwargs):

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
    
    if dataset == 'mnist' or dataset == 'fashion':

        x, _, _, _ = load_mnist(dataset, kwargs['n'], kwargs['downsample_factor'],kwargs['whiten'], device)

        # calculate the empirical input covariance matrix and its condition number
        cov_xx = (x.T @ x / kwargs['n']).to(device)
        mat_rank = torch.linalg.matrix_rank(cov_xx)
        print(mat_rank)

        eigvals = torch.linalg.eigvalsh(cov_xx)
#         cond_cov_xx = float(torch.linalg.cond(cov_xx))
        cond_cov_xx = float(eigvals[-1]/eigvals[-mat_rank])

        print('Condition number of MNIST dataset: %.3f' %cond_cov_xx)
        
    elif dataset == 'cifar-10':
        
        x, _, _, _ = load_cifar10(kwargs['n'], kwargs['grayscale'], kwargs['flatten'], kwargs['whiten'], device)

        n = x.shape[0]
        # calculate the empirical input covariance matrix and its condition number
        cov_xx = (x.T @ x / n).to(device)
        mat_rank = torch.linalg.matrix_rank(cov_xx)
        print(mat_rank)

        eigvals = torch.linalg.eigvalsh(cov_xx)
#         cond_cov_xx = float(torch.linalg.cond(cov_xx))
        cond_cov_xx = float(eigvals[-1]/eigvals[-mat_rank])

        print('Condition number of Cifar-10 dataset: %.3f' %cond_cov_xx)
    elif dataset == 'gaussian':

        # # # sample n (input,output) tuples
        n = config['data_pts_per_class_training']

        x = kwargs['data']
        # calculate the empirical input covariance matrix and its condition number
        cov_xx = torch.tensor(x.T @ x / n).to(device)

        # mat_rank = torch.linalg.matrix_rank(cov_xx)
        # eigvals = torch.sort(abs(torch.linalg.eigvalsh(cov_xx))).values
        cond_cov_xx = float(torch.linalg.cond(cov_xx))
        # cond_cov_xx = eigvals[-1]/eigvals[0]


        print('Condition number of bimodal gaussian dataset: %.3f' %cond_cov_xx)

    else:
        ValueError('Unknown dataset chosen.')


    print('Initializing Networks...')


    print('Number of networks:', len(networks))

    for network in networks:
        
        time_start = datetime.datetime.now()
            
        
        d = network.input_dim
        k = network.output_dim
        l = network.L
        m = network.width

        print('Network configuration: d=%d, k=%d, m=%d, L=%d' % (d,k, network.width, l+1))

        torch.manual_seed(seed)

        for i in range(num_inits):

            # initialize the weight matrices according to the defined initialization
            network.init_weights(init_type)
            
            # alpha = network.activ.negative_slope
            
            # calculate upper bounds

            if config['model_name'] == 'sequential' and config['activation_func'] == 'linear':
                
                H_o_cond, H_o_cond_bound1, H_o_cond_bound2, H_o_cond_bound3 = calc_cond_num_outer_prod_hess_explicit_linearNN(d, l, k, m, network, cond_cov_xx, cov_xx, device)
            elif config['model_name'] == 'lin_residual_network':
                H_o_cond, H_o_cond_bound1, H_o_cond_bound2, H_o_cond_bound3 = calc_cond_num_outer_prod_hess_explicit_linearResNN(d, l, k, m, network, cond_cov_xx, cov_xx, device)
            elif config['model_name'] == 'sequential' and config['activation_func'] == 'leaky_relu':

                H_o_cond, H_o_cond_bound1, H_o_cond_bound2, H_o_cond_bound15 = calc_cond_num_outer_prod_hess_explicit_1_hiddenlayer_leaky_ReLU(network, x.T)

            outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                       cond_cov_xx,
                                                                network.input_dim, network.output_dim, 
                                                                network.width, network.depth, network.activation_func, 0,
                                                                'H_o_cond', H_o_cond
                                                                ]
            outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                       cond_cov_xx,
                                                                network.input_dim, network.output_dim, 
                                                                network.width, network.depth, network.activation_func, 0,
                                                                'H_o_cond_bound1', H_o_cond_bound1
                                                                ]
            outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                       cond_cov_xx,
                                                                network.input_dim, network.output_dim, 
                                                                network.width, network.depth, network.activation_func, 0,
                                                                'H_o_cond_bound2', H_o_cond_bound2
                                                                ]
            outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                       cond_cov_xx,
                                                                network.input_dim, network.output_dim, 
                                                                network.width, network.depth, network.activation_func, 0,
                                                                'H_o_cond_bound1.5', H_o_cond_bound15
                                                                ]
        print('Time passed for H_O calculations: %.3f seconds' %(datetime.datetime.now()-time_start).total_seconds())  
    
    return outer_prod_hessian_information
    

def main(project_name, experiment_name, config):

    if config['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    print('device: %s' %device)
        
    print('Running experiment %s for project %s ' %(experiment_name, project_name))

    # setting up Gaussian dataset
    ds = config['input_dim']
    ks = config['output_dim']

    torch.manual_seed(config['seed'])

    n = config['data_pts_per_class_training']
    m = config['data_pts_per_class_validation']
    v = config['data_pts_per_class_testing']

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

    for d in ds:
        for k in ks:

            time_start = datetime.datetime.now()
            
            # Set network configurations (input dimension, output dimension, hidden units, num_hidden_layers etc.)
            # for both linear and ReLU networks
            
            m1 = config['width'] 
            activ_func = config['activation_func']

            num_inits = config['num_initialization']

            L = np.array(config['hidden_layers'])-1 # number of hidden layers of dim "m"   
            
            init_type = config['init_type']
                
            m_L_config = [] # keep track of the network configuration
            num_param = [] # count the number of parameters in the model

            Networks = [] # list of NN with different configurations


            # initiate networks of given depth L[l] with m hidden units each
            for m in m1:
                for l in L:
                    m_L_config.append((m,l))
                    if config['model_name'] == 'sequential':

                        if config['activation_func'] == 'leaky_relu':
                            kwargs={'neg_slope': config['neg_slope'], 'batch_norm': config['batch_norm']} # negative slope of leaky ReLU
                            Networks.append(Sequential_NN(d,m,k,l,'leaky_relu',**kwargs).to(device))
                        else:
                            kwargs={'batch_norm': config['batch_norm']}
                            Networks.append(Sequential_NN(d,m,k,l,activ_func, **kwargs).to(device))
                    elif config['model_name'] == 'sequential_w_fully_skip':
                        if config['beta'] == '1/sqrt(L)':
                            beta = 1/np.sqrt(l+1)
                        elif config['beta'] == '1/L':
                            beta = 1/(l+1)
                        else:
                            beta = config['beta']
                        if config['activation_func'] == 'leaky_relu':
                            kwargs={'neg_slope': config['neg_slope']} # negative slope of leaky ReLU
                            Networks.append(Sequential_fully_skip_NN(d,m,k,l,beta=beta,
                                                                    activation=activ_func,**kwargs).to(device))
                        else: 
                            Networks.append(Sequential_fully_skip_NN(d,m,k,l,beta=beta,
                                                                    activation=activ_func).to(device))
                    elif config['model_name'] == 'lin_residual_network':
                        if config['beta'] == '1/sqrt(L)':
                            beta = 1/np.sqrt(l+1)
                        elif config['beta'] == '1/L':
                            beta = 1/(l+1)
                        else:
                            beta = config['beta']
                        print('beta=',beta)
                        Networks.append(Linear_skip_single_layer_NN(d,m,k,l,beta=beta).to(device))
                    else:
                        ValueError('Unknown model_name in config file')

                    num_param.append(sum(p.numel() for p in Sequential_fully_skip_NN(d,m,k,l).parameters()))
                    
            print('num parameters: ', num_param)


            if config['loss_func'] == 'mse':
                loss_func = F.mse_loss

                if config['dataset'] == 'mnist' or config['dataset'] == 'fashion':
                    n = config['datapoints']
                    x_train, y_train, _, _ = load_mnist(config['dataset'], n, config['downsample_factor'],config['whiten_mnist'], device)

                    # calculate the empirical input covariance matrix and its condition number
                    cov_xx = (x_train.T @ x_train / n).to(device)
                    mat_rank = torch.linalg.matrix_rank(cov_xx)
                    print(mat_rank)

                    eigvals = torch.linalg.eigvalsh(cov_xx)
                    #         cond_cov_xx = float(torch.linalg.cond(cov_xx))
                    cond_cov_xx = float(eigvals[-1]/eigvals[-mat_rank])

                    print('Condition number of MNIST dataset: %.3f' %cond_cov_xx)

                elif config['dataset'] == 'cifar-10':

                    x_train, y_train, _, _ = load_cifar10(config['datapoints_cifar10'], config['grayscale'], config['flatten'], config['whiten_cifar10'], device)

                    n = x_train.shape[0]
                    # calculate the empirical input covariance matrix and its condition number
                    cov_xx = (x_train.T @ x_train / n).to(device)
                    mat_rank = torch.linalg.matrix_rank(cov_xx)
                    print(mat_rank)

                    eigvals = torch.linalg.eigvalsh(cov_xx)
                    #         cond_cov_xx = float(torch.linalg.cond(cov_xx))
                    cond_cov_xx = float(eigvals[-1]/eigvals[-mat_rank])

                    print('Condition number of Cifar-10 dataset: %.3f' %cond_cov_xx)
                elif config['dataset'] == 'gaussian':

                    # # # sample n (input,output) tuples
                    n = config['data_pts_per_class_training']
                    # # x = mvrn_d.rsample((n,))    
                    # # y = mvrn_k.rsample((n,))
                    x_train, y_train = generate_gaussian_data_for_bin_classification(n, d, k, device)

                    # calculate the empirical input covariance matrix and its condition number
                    cov_xx = torch.tensor(x_train.T @ x_train / n).to(device)

                    # mat_rank = torch.linalg.matrix_rank(cov_xx)
                    # eigvals = torch.sort(abs(torch.linalg.eigvalsh(cov_xx))).values
                    cond_cov_xx = float(torch.linalg.cond(cov_xx))
                    # cond_cov_xx = eigvals[-1]/eigvals[0]

                if config['calc_hess_info'] == True:
                    _hessian_information  = eval_network_configurations(num_inits, init_type, Networks, x_train, y_train, loss_func, m_L_config, calc_H=config['calc_H'],
                                                                            method_cond_num=config['method_cond_num'],
                                                                            seed=config['seed'],device=device)
                
                if config['calc_H_O_info'] == True:
                    if config['dataset'] == 'gaussian':
                        _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(num_inits, init_type, Networks, 'gaussian', seed=314159, device=device, data=x_train, X=x_train.T)
                    elif config['dataset'] == 'mnist':
                        if config['activation_func'] == 'leaky_relu':
                            _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(num_inits, init_type, Networks, 'mnist', seed=314159, device=device, n=config['datapoints'],downsample_factor=config['downsample_factor'], whiten=config['whiten_mnist'])
                        else:
                            _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(num_inits, init_type, Networks, 'mnist', seed=314159, device=device, n=config['datapoints'],downsample_factor=config['downsample_factor'], whiten=config['whiten_mnist'])
                    elif config['dataset'] == 'cifar-10':
                        _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(num_inits, init_type, Networks, 'cifar-10', seed=314159, device=device, grayscale=config['grayscale'], n=config['datapoints_cifar10'], flatten=config['flatten'], whiten=config['whiten_cifar10'])

                
            if config['calc_hess_info'] == True:
                hessian_information = pd.concat([hessian_information, _hessian_information],ignore_index=True)

            if config['calc_H_O_info'] == True:
                outer_prod_hessian_information = pd.concat([outer_prod_hessian_information, _outer_prod_hessian_information],ignore_index=True)

            print('Time passed: %.3f seconds' %(datetime.datetime.now()-time_start).total_seconds())         
    time_now = dt.now().isoformat()

    if config['log_yaml_file']  == True:
        
        file_path = 'figures/' + config['project_name'] + '_' + config['experiment_name'] + '_' + time_now + '/'
        os.mkdir(file_path)
    
        yaml_file_name = file_path + 'config_' + time_now + '.yaml'
        with open(yaml_file_name, 'w') as file:
            yaml.dump(config, file)
       
    if config['calc_hess_info'] == True:
        hessian_information.to_pickle("pandas_dataframes/hessian_information_%s_%s.pkl" %(config['project_name'], config['experiment_name']))
    if config['calc_H_O_info'] == True:
        outer_prod_hessian_information.to_pickle("pandas_dataframes/outer_prod_hessian_information_%s_%s.pkl" %(config['project_name'], config['experiment_name']))

if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)

    plt.rcParams["figure.figsize"] = (4,4)

    # load in YAML configuration
    config = {}
    base_config_path = 'config_init_experiments.yaml'
    with open(base_config_path, 'r') as file:
        config.update(yaml.safe_load(file))


    # start training with name and config 
    main(config['project_name'], config['experiment_name'], config)

