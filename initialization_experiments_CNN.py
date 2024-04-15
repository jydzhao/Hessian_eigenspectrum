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
import scipy



def calc_cond_num_outer_prod_hess_explicit_linear_CNN(network, cond_cov_xx, cov_xx, device):
    
    # first generate the Toeplitz matrices from the convolutional filters
    
    
    d = cov_xx.shape[0]
    L = network.depth
    m = network.num_channels
    k = network.kernel_size
    
    ks = network.kernel_size
    
    d_ls = np.array([d-i*(ks-1) for i in range(L+2)])

    m_ls = np.zeros(L+2,dtype=np.int64)
    m_ls[0] = 1
    m_ls[1:] = m
    
    T_ls = []
    for l in range(L+1):
        T_l = np.zeros((m_ls[l+1]*d_ls[l+1],m_ls[l]*d_ls[l]))
        T_ls.append(T_l)
        
    # fill Toeplitz matrices
    for l in range(L+1):
        if l == 0:
            Wl = network.conv_in.weight.detach()
        else:
            Wl = network.conv_hidden[l-1].weight.detach()

        T_l = T_ls[l]
        
        d_l = d_ls[l+1]
        d_l_m1 = d_ls[l]

        for i in range(m_ls[l+1]):
            for j in range(m_ls[l]):
                Wl_ij = Wl[i,j,:]

                

                d1 = np.zeros(d_l)
                d2 = np.zeros(d_l_m1)

                d1[0] = Wl_ij[0]
                d2[0:ks] = Wl_ij
                T_Wl_ij = scipy.linalg.toeplitz(d1, d2)

                T_l[i*d_l:(i+1)*d_l,j*d_l_m1:(j+1)*d_l_m1] = T_Wl_ij
                
    # output of CNN is T_l[L] @ T_l[L-1] @ ... @ T_l[0] @ x
    

    # calculate the theoretical upper bound of the condition number of outer product Hessian 
    # based on the first approximation 
    lam_max_upperb = 0
    lam_min_lowerb = 0
    
    l = L-1
    
    for el in range(1,l+2+1):
        if el == l+2:
            W_1 = torch.eye(k) 
        else:
            W_1 = torch.tensor(T_ls[1])
        
            for lay in range(-1,-2-l+el,-1):
                W_1 = W_1 @ network.lin_hidden[lay].weight.detach()
        
        if el == 1:
            W_2 = torch.eye(d)
        else:
            W_2 = torch.tensor(T_ls[0])
            
            for lay in range(el-2):
                W_2 = W_2 @ network.lin_hidden[lay].weight.detach().T
        
        lam_max_upperb += torch.linalg.eigvalsh(W_1 @ W_1.T)[-1] * torch.linalg.eigvalsh(W_2 @ W_2.T)[-1]
        lam_min_lowerb += torch.linalg.eigvalsh(W_1 @ W_1.T)[0] * torch.linalg.eigvalsh(W_2 @ W_2.T)[0]            
        
    cond_H_o_tilde_upper_bound_Eq5657 = ((lam_max_upperb)/(lam_min_lowerb)) * cond_cov_xx


    # calculate the outer Hessian product according to the expression derived by Sidak (Eq. 2)
    UT_U = torch.zeros((k*d,k*d)).to(device)
    
    I_out = torch.eye(d_ls[-1]*m_ls[-1]).to(device)
    I_d = torch.eye(d).to(device)
    
    if l == 0:
        V_kaiming = torch.tensor(T_ls[0])
        W_kaiming = torch.tensor(T_ls[1])
        
#         print(cov_xx.device)
        
        print(V_kaiming.shape,W_kaiming.shape)
  
        H_o_tilde = torch.kron(W_kaiming@W_kaiming.T, cov_xx) + torch.kron( I_out, cov_xx**1/2 @ V_kaiming.T@V_kaiming@cov_xx**1/2)

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

    return float(cond_H_o_tilde), float(cond_H_o_tilde_upper_bound_Eq5657)
   

def eval_network_cond_num_outer_prod_hess(num_inits, init_type, networks, dataset, seed=314159, device='cpu', **kwargs):

    outer_prod_hessian_information = pd.DataFrame({'dataset':[],
                                        'network':[],
                                        'cond_cov_xx':[],
                                        'input_dim':[],
                                        'kernel_size':[],
                                        'num_filters':[],
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
            
        
#         d = network.input_dim
#         k = network.output_dim
#         l = network.L
#         m = network.width

        print('Network configuration: m=%d, kernel_size=%d, L=%d' % (network.num_channels, network.kernel_size, network.depth+1))

        torch.manual_seed(seed)

        for i in range(num_inits):

            # initialize the weight matrices according to the defined initialization
            network.init_weights(init_type)
            
            # alpha = network.activ.negative_slope
            
            # calculate upper bounds

            if config['model_name'] == 'sequential_cnn' and config['activation_func'] == 'linear':
                
                H_o_cond, H_o_cond_bound1 = calc_cond_num_outer_prod_hess_explicit_linear_CNN(network, cond_cov_xx, cov_xx, device)
            

            outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                       cond_cov_xx,
                                                                cov_xx.shape[0], network.kernel_size,
                                                                network.num_channels, network.depth, network.activation_func, 0,
                                                                'H_o_cond', H_o_cond
                                                                ]
            outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                       cond_cov_xx,
                                                                cov_xx.shape[0], network.kernel_size,
                                                                network.num_channels, network.depth, network.activation_func, 0,
                                                                'H_o_cond_bound1', H_o_cond_bound1
                                                                ]
#            
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

    outer_prod_hessian_information = pd.DataFrame({'dataset':[],
                                        'network':[],
                                        'cond_cov_xx':[],
                                        'input_dim':[],
                                        'kernel_size':[],
                                        'num_filters':[],
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
            
            m1 = config['filter'] 
            activ_func = config['activation_func']

            num_inits = config['num_initialization']
            
            kernel_sizes = config['kernel_size']

            L = np.array(config['hidden_layers']) # number of hidden layers of dim "m"   
            
            init_type = config['init_type']
                
            m_L_config = [] # keep track of the network configuration
            num_param = [] # count the number of parameters in the model

            Networks = [] # list of NN with different configurations

            
            # initiate networks of given depth L[l] with m hidden units each
            for m in m1:
                for l in L:
                    for kernel_size in kernel_sizes:
                        m_L_config.append((m,l))
                        if config['model_name'] == 'sequential_cnn':

                            if config['activation_func'] == 'leaky_relu':
                                kwargs={'neg_slope': config['neg_slope'], 'batch_norm': config['batch_norm']} # negative slope of leaky ReLU
                                Networks.append(Sequential_CNN(m,l,kernel_size,'leaky_relu',**kwargs).to(device))
                            else:
                                kwargs={'batch_norm': config['batch_norm']}
                                Networks.append(Sequential_CNN(m,l,kernel_size,activ_func, **kwargs).to(device))

                        else:
                            ValueError('Unknown model_name in config file')

                        num_param.append(sum(p.numel() for p in Sequential_CNN(m,l,kernel_size,**kwargs).parameters()))
                    
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
                
                
                
                if config['calc_H_O_info'] == True:
                    if config['dataset'] == 'mnist':
                        if config['activation_func'] == 'leaky_relu':
                            _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(num_inits, init_type, Networks, 'mnist', seed=314159, device=device, n=config['datapoints'],downsample_factor=config['downsample_factor'], whiten=config['whiten_mnist'])
                        else:
                            _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(num_inits, init_type, Networks, 'mnist', seed=314159, device=device, n=config['datapoints'],downsample_factor=config['downsample_factor'], whiten=config['whiten_mnist'])
                    elif config['dataset'] == 'cifar-10':
                        _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(num_inits, init_type, Networks, 'cifar-10', seed=314159, device=device, grayscale=config['grayscale'], n=config['datapoints_cifar10'], flatten=config['flatten'], whiten=config['whiten_cifar10'])

                

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
    base_config_path = 'config_init_experiments_CNN.yaml'
    with open(base_config_path, 'r') as file:
        config.update(yaml.safe_load(file))


    # start training with name and config 
    main(config['project_name'], config['experiment_name'], config)

