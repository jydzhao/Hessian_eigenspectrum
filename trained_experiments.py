import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
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
        V_kaiming = network.lin_in.weight.detach().to(device)
        W_kaiming = network.lin_out.weight.detach().to(device)

        print(W_kaiming.device, cov_xx.device)
            
        H_o_tilde = torch.kron(W_kaiming@W_kaiming.T, cov_xx) + torch.kron( torch.eye(k), V_kaiming.T@V_kaiming@cov_xx)

        mat_rank = torch.linalg.matrix_rank(H_o_tilde, atol=1e-7)
        eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
        cond_H_o_tilde = torch.linalg.cond(H_o_tilde)
#         print(mat_rank)
#         cond_H_o_tilde = eigvals[-1]/eigvals[-mat_rank]
    

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

        H_o_tilde = UT_U @ (torch.kron(torch.eye(k), cov_xx))

        mat_rank = torch.linalg.matrix_rank(H_o_tilde, atol=1e-8)
        eigvals = torch.sort(abs(torch.linalg.eigvalsh(H_o_tilde))).values
        print(mat_rank)
        # print(len(eigvals))
        # print(eigvals)
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

def eval_network_cond_num_outer_prod_hess(networks, dataset, device='cpu', **kwargs):

    
    
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

        x, _, _, _, = load_mnist(dataset, kwargs['n'], kwargs['downsample_factor'],kwargs['whiten'], device)

        # calculate the empirical input covariance matrix and its condition number
        cov_xx = x.T @ x / kwargs['n']
        cov_xx = cov_xx.to(device)
        
        mat_rank = torch.linalg.matrix_rank(cov_xx)
        print('n=',kwargs['n'])
        print('rank cov_xx: ', mat_rank)

        eigvals = torch.linalg.eigvalsh(cov_xx)
        # cond_cov_xx = float(torch.linalg.cond(cov_xx))
        cond_cov_xx = float(eigvals[-1]/eigvals[-mat_rank])

    for ind, network in enumerate(networks):
        
        d = network.input_dim
        k = network.output_dim
        l = network.L
        m = network.width

        print('Network configuration: d=%d, k=%d, m=%d, L=%d' % (d,k, m, l))

        if dataset == 'gaussian':

           
            n = config['data_pts_per_class_training']
           
            x, _ = generate_gaussian_data_for_bin_classification(n, d, k, config['seed'], device)

            cov_xx = x.T @ x / n
            
            cov_xx = cov_xx.to(device)

            # mat_rank = torch.linalg.matrix_rank(cov_xx)
            # eigvals = torch.sort(abs(torch.linalg.eigvalsh(cov_xx))).values
            cond_cov_xx = float(torch.linalg.cond(cov_xx))
            # cond_cov_xx = eigvals[-1]/eigvals[0]
        elif dataset == 'mnist' or dataset == 'fashion':
            print(dataset, ' dataset')
            
#             x,_ ,_, _ = load_mnist(dataset, n, kwargs['downsample_factor'], kwargs['whiten'], device)
        else:
            ValueError('Unknown dataset chosen.')
            
        # calculate upper bounds

        if config['model_name'] == 'sequential' and network.activation_func == 'linear':
            H_o_cond, H_o_cond_bound1, H_o_cond_bound2, H_o_cond_bound3 = calc_cond_num_outer_prod_hess_explicit_linearNN(d, l, k, m, network, cond_cov_xx, cov_xx)
        elif config['model_name'] == 'lin_residual_network':
            H_o_cond, H_o_cond_bound1, H_o_cond_bound2, H_o_cond_bound3 = calc_cond_num_outer_prod_hess_explicit_linearResNN(d, l, k, m, network, cond_cov_xx, cov_xx)


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
        outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = [dataset, config['model_name'],
                                                                                    cond_cov_xx,
                                                            network.input_dim, network.output_dim, 
                                                           network.width, network.depth, network.activation_func,
                                                                                   kwargs['epochs'][ind],
                                                            'H_o_cond_bound3', H_o_cond_bound3
                                                            ]
    
    return outer_prod_hessian_information
    


def main(project_name, experiment_name, config):

    device = torch.device('cpu')
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Running experiment %s for project %s ' %(experiment_name, project_name))
    
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


    time_start = datetime.datetime.now()
        

    Networks = [] # list of NN with different configurations

    epochs = config['epochs']
    for filename in config['filenames']:
        for epoch in epochs:
            filepath = config['path']+filename + 'epoch=%d' %epoch + '.pt' 
            
            network = torch.load(filepath)
        # network = torch.jit.load(config['path']+filename)
        # network.eval()
        
            Networks.append(network)

    if config['loss_func'] == 'mse':
        loss_func = F.mse_loss

        # if config['calc_H_O_info'] == True:
        if config['dataset'] == 'gaussian':
            _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(Networks, 'gaussian', seed=314159)
        elif config['dataset'] == 'mnist':
            _outer_prod_hessian_information = eval_network_cond_num_outer_prod_hess(Networks, 'mnist', seed=314159, 
                                                                                    n=config['datapoints'], 
                                                                                    downsample_factor=config['downsample_factor'],device=device,
                                                                                    whiten=config['whiten_mnist'], epochs=epochs)

    # if config['calc_H_O_info'] == True:
    outer_prod_hessian_information = pd.concat([outer_prod_hessian_information, _outer_prod_hessian_information],ignore_index=True)

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

if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)

    plt.rcParams["figure.figsize"] = (4,4)

    # load in YAML configuration
    config = {}
    base_config_path = 'config_trained_experiments.yaml'
    with open(base_config_path, 'r') as file:
        config.update(yaml.safe_load(file))

    # # TODO: add more if more parameters should be "sweepable"
    # # overwrite with sweep parameters - have to be given in with ArgumentParser for wandb
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--L2_clip', type=float, default=config['L2_clip'], help='L2 clip for DP')
    # args = parser.parse_args()

    # # TODO: check for easy way to convert args to dict to simply update config
    # config['L2_clip'] = args.L2_clip
    # config['max_epochs'] = args.max_epochs

    # start training with name and config 
    main(config['project_name'], config['experiment_name'], config)

