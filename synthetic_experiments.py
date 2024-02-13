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

import mnist_reader


def train_network(x_train, y_train, train_dl, loss_func, network, optimizer, lr, epochs, calc_cond_num=False, method_cond_num='naive', verbose_level=0, calc_every_x_epoch=10, device='cpu',save_model=False, **kwargs):
    
    print('calc_every_x_epoch',calc_every_x_epoch)
    
    if optimizer == 'SGD':
        opt = optim.SGD(network.parameters(), lr=lr)
    elif optimizer == 'Adam':
        opt = optim.Adam(network.parameters(), lr=lr)
    else:
        ValueError('Unknown optimizer')    
    

    training_information = pd.DataFrame({
                                    'width':[],
                                    'depth':[],
                                    'activ_f':[],
                                    'epoch':[],
                                    'loss':[],
                                    'grad_norm_squared':[]
                                    })
    
    initial_loss = loss_func(network(x_train), y_train)
    initial_loss.backward()
    grad_norm_sq = sum([torch.linalg.norm(param.grad)**2 for param in network.parameters()])
    opt.zero_grad()

    training_information.loc[len(training_information)] = [network.width, network.depth, network.activation_func, 0, loss_func(network(x_train), y_train).cpu().detach(), grad_norm_sq.cpu().detach()]
    
    if calc_cond_num == True:
        hessian_information = pd.DataFrame({ 
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
 
    if calc_cond_num == True:
        _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _lam_abs_min_H_o, _lam_abs_max_H_o, _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o, _H_rank, _H_o_rank = calc_condition_num(network,
                                                                    x_train,y_train,
                                                                    loss_func,
                                                                    method_cond_num)
        
        hessian_information.loc[len(hessian_information)] = [network.width, network.depth, network.activation_func, 0, 
                                                             _H_cond, _H_o_cond, 
                                                             _lam_abs_min_H, _lam_abs_max_H, 
                                                             _lam_abs_min_H_o, _lam_abs_max_H_o,
                                                             _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o,
                                                             _H_rank, _H_o_rank]
        
    print('Epoch: 0 \t loss= %10.3e' %loss_func(network(x_train), y_train).detach())

    for epoch in tqdm(range(epochs)):
        
        if save_model == True and epoch%calc_every_x_epoch==0:
            # print(network.lin_in.weight)
            filename = (kwargs['save_path'] + 'network_' + 'd=%d_' + 'm=%d_' + 'k=%d_' + 'L=%d_' + '%s_' + '%s_' + '%s' + '_epoch=%s' + '.pt') % (network.input_dim, network.width, network.output_dim, network.depth, network.activation_func, optimizer, kwargs['dataset'], epoch)
            # model_scripted.save(filename)
            torch.save(network, filename)
        
        if calc_cond_num == True and epoch%calc_every_x_epoch==0:
            _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _lam_abs_min_H_o, _lam_abs_max_H_o, _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o, _H_rank, _H_o_rank = calc_condition_num(network,
                                                                                                                                                                            x_train,y_train,
                                                                                                                                                                            loss_func,
                                                                                                                                                                            method_cond_num)

            hessian_information.loc[len(hessian_information)] = [network.width, network.depth, network.activation_func, epoch, 
                                                                _H_cond, _H_o_cond,
                                                                _lam_abs_min_H, _lam_abs_max_H, 
                                                                _lam_abs_min_H_o, _lam_abs_max_H_o,
                                                                _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o,
                                                                _H_rank, _H_o_rank]           
        
        for xb, yb in train_dl:
            pred = network(xb)        
            loss = loss_func(pred, yb, reduction='mean')

            loss.backward()
            
            grad_norm_sq = sum([torch.linalg.norm(param.grad)**2 for param in network.parameters()])

            training_information.loc[len(training_information)] = [network.width, network.depth, network.activation_func, epoch, loss_func(network(x_train), y_train).cpu().detach(), grad_norm_sq.cpu().detach()]
            
            opt.step()
            opt.zero_grad()

            if verbose_level >= 2:
                print('loss: ', loss_func(network(x_train), y_train).detach())
                    
                print('gradient_norm_sqaured: ', grad_norm_sq)

        
        
        
        if verbose_level == 0 and epoch%int(epochs/10 + 1) ==0:
            print('Epoch: %d \t loss= %10.4e' %(epoch+1, loss_func(network(x_train), y_train).detach()))
        elif verbose_level >= 1:
            print('Epoch: %d \t loss= %10.4e' %(epoch+1, loss_func(network(x_train), y_train).detach()))
         
        
    print('Epoch: %d \t loss= %10.4e' %(epoch+1, loss_func(network(x_train), y_train).detach()))

    if save_model == True:
        # print(network.lin_in.weight)
        filename = (kwargs['save_path'] + 'network_' + 'd=%d_' + 'm=%d_' + 'k=%d_' + 'L=%d_' + '%s_' + '%s_' + '%s_' + 'epoch=%s' + '.pt') % (network.input_dim, network.width, network.output_dim, network.depth, network.activation_func, optimizer, kwargs['dataset'], epoch)
        # model_scripted.save(filename)
        torch.save(network, filename)
        
        
    if calc_cond_num == True:
        return training_information, hessian_information
    else:
        return training_information

def train_network_configurations(networks, x_train, y_train, train_dl, loss_func, m_L_config, epochs, lrs, optimizer='SGD', calc_cond_num=False, method_cond_num='naive', calc_every_x_epoch=10, verbose_level=0, seed=314159, device='cpu', save_model=False, **kwargs):
    '''
    Train list of networks of different "configurations" (#hidden units, #hidden layers)
    networks: List of networks to be trained
    m_L_config: List of configurations for each network (only needed to print information on the networks)
    epochs: number of epochs to be trained
    lrs: fixed learning-rate for each of the network
    optimizer: 'SGD' or 'Adam' optimizer
    calc_condition_num: Boolean whether to calculate  the condition number of the Hessian and the eigenvalues 
    calc_every_x_epoch: Calculates the condition number of the Hessian and the eigenvalues every x epochs
    verbose_level: 0,1,2, to track loss, mostly for debugging
    seed: seed for initializing the network according to Kaiming normal initialization
    
    '''

    # Train list of networks

    # Define DataFrames to log information about training process and Hessian information (condition number, eigenvalues etc.)
    training_information = pd.DataFrame({
                                    'width':[],
                                    'depth':[],
                                    'activ_f':[],
                                    'epoch':[],
                                    'loss':[],
                                    'grad_norm_squared':[]
                                    })
    if calc_cond_num == True:
        hessian_information = pd.DataFrame({ 
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
                                            'std_diff_H_H_o':[]
                                            })
    


    print('Training Networks...')

    for ind, network in enumerate(networks):

        print('Network configuration: d=%d, k=%d, m=%d, L=%d' % (networks[0].lin_in.weight.shape[1], networks[0].lin_out.weight.shape[0], m_L_config[ind][0], m_L_config[ind][1]+1))

        # torch.manual_seed(seed)

        network.init_weights('kaiming_normal')

        if np.isscalar(lrs):
            lr = lrs
        else:
            try:
                lr = lrs[ind]
            except:
                TypeError('Unknown format for learning rate')
        
        if calc_cond_num==False:
            if save_model == True:
                
                _training_information = train_network(x_train, y_train, train_dl,
                                                loss_func, network, 
                                                optimizer=optimizer, 
                                                lr=lr, epochs=epochs, 
                                                calc_cond_num=False,
                                                verbose_level=verbose_level,
                                                device=device, save_model=save_model, save_path= kwargs['save_path'], dataset= kwargs['dataset'])
            else:
                _training_information = train_network(x_train, y_train, train_dl,
                                                loss_func, network, 
                                                optimizer=optimizer, 
                                                lr=lr, epochs=epochs, 
                                                calc_cond_num=False,
                                                verbose_level=verbose_level,
                                                device=device)
            training_information = pd.concat([training_information, _training_information])
        else: 
            _training_information, _hessian_information = train_network(x_train, y_train, train_dl, 
                                                                        loss_func, network, 
                                                                        optimizer=optimizer, 
                                                                        lr=lr, epochs=epochs, 
                                                                        calc_cond_num=True, 
                                                                        method_cond_num=method_cond_num, 
                                                                        verbose_level=verbose_level,
                                                                        calc_every_x_epoch=calc_every_x_epoch,
                                                                        device=device)
            
            training_information = pd.concat([training_information, _training_information],ignore_index=True)
            hessian_information = pd.concat([hessian_information, _hessian_information],ignore_index=True)

    training_information['grad_norm_squared'] = training_information['grad_norm_squared'].astype(float)
    training_information['loss'] = training_information['loss'].astype(float)
        
    if calc_cond_num==False:
        
        return training_information
    else: 
        
        hessian_information['H_o_cond'] = hessian_information['H_o_cond'].astype(float)
        hessian_information['mean_diff_H_H_o'] = hessian_information['mean_diff_H_H_o'].astype(float)
        hessian_information['max_diff_H_H_o'] = hessian_information['max_diff_H_H_o'].astype(float)
        hessian_information['std_diff_H_H_o'] = hessian_information['std_diff_H_H_o'].astype(float)

        return training_information, hessian_information



def main(project_name, experiment_name, config):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    print('Running experiment %s for project %s ' %(experiment_name, project_name))

    if config['dataset'] == 'gaussian':
    # setting up Gaussian dataset

        torch.manual_seed(config['seed'])

        n = config['data_pts_per_class_training']
        m = config['data_pts_per_class_validation']
        v = config['data_pts_per_class_testing']
        
        d = config['input_dim']
        k = config['output_dim']

        x_train, y_train = generate_gaussian_data_for_bin_classification(n, d, k, config['whiten_gaussian'], config['seed'], device)
        x_test, y_test = generate_gaussian_data_for_bin_classification(m, d, k, config['whiten_gaussian'], config['seed'], device)
        x_val, y_val = generate_gaussian_data_for_bin_classification(v, d, k, config['whiten_gaussian'], config['seed'], device)
        
        if config['whiten_gaussian']:
            filename = project_name + '_' + experiment_name + '_x_train' + 'whitened.npy'
            save_run(filename, x_train.cpu())
            
        else:
            filename = project_name + '_' + experiment_name + '_x_train' + 'NOTwhitened.npy'
            save_run(filename, x_train.cpu())
            
    elif config['dataset'] == 'mnist' or config['dataset'] == 'fashion':
        x_train, y_train, x_val, y_val, x_test, y_test = load_mnist(config['dataset'] ,config['datapoints'], config['downsample_factor'], config['normalize_mnist'], device=config['device'])

        
    train_dl, valid_dl = create_dataloaders(x_train, y_train, x_val, y_val, config['batch_size'])

    # calculate the empirical input covariance matrix and its condition number
    cov_xx = x_train.T @ x_train / len(x_train)

    cov_xx_rank = torch.linalg.matrix_rank(cov_xx,atol=1e-15)
    spectrum = torch.linalg.eigvalsh(cov_xx)

    print('Rank of input covariance matrix: %d ' % cov_xx_rank)

    print('Condition number of input covariance matrix: %.4f ' % (spectrum[-1]/spectrum[-cov_xx_rank]) )


    # Set network configurations (input dimension, output dimension, hidden units, num_hidden_layers etc.)
    # for both linear and ReLU networks

    d = config['input_dim']
    m1 = config['width']
    k = config['output_dim'] # output dimension
    activ_func = config['activation_func']

    L = np.array(config['hidden_layers'])-1 # number of hidden layers of dim "m"       
        
    m_L_config = [] # keep track of the network configuration
    num_param = [] # count the number of parameters in the model

    Networks = [] # list of NN with different configurations

    # Parameters for plotting
    hue_var = config['hue_var']
    size_var = config['size_var']

    print('Initiate networks...')

    # initiate networks of given depth L[l] with m1 hidden units each
    for m in m1:
        for l in L:
            m_L_config.append((m,l))
            if config['model_name'] == 'sequential':

                if config['activation_func'] == 'leaky_relu':
                    kwargs={'neg_slope': config['neg_slope'], 'batch_norm':config['batch_norm']} # negative slope of leaky ReLU
                    Networks.append(Sequential_NN(d,m,k,l,'leaky_relu',**kwargs).to(device))
                else:
                    # print(d,m,k,l)
                    # print(activ_func)
                    kwargs={'batch_norm':config['batch_norm']}
                    Networks.append(Sequential_NN(d,m,k,l,activ_func,**kwargs).to(device))
            elif config['model_name'] == 'sequential_w_fully_skip':
                if config['activation_func'] == 'leaky_relu':
                    kwargs={'neg_slope': config['neg_slope']} # negative slope of leaky ReLU
                    Networks.append(Sequential_fully_skip_NN(d,m,k,l,beta=config['beta'],
                                                            activation=activ_func,**kwargs).to(device))
                else: 
                    Networks.append(Sequential_fully_skip_NN(d,m,k,l,beta=config['beta'],
                                                            activation='linear').to(device))
            else:
                ValueError('Unknown model_name in config file')

            num_param.append(sum(p.numel() for p in Sequential_fully_skip_NN(d,m,k,l).parameters()))
            
    print('num parameters: ', num_param)

    # Train Networks

    epochs = config['max_epochs']
    calc_every_x_epoch = config['calc_every_iter']
    # lrs = [0.02, 0.01, 0.01, 0.001, 0.0006, 0.0003] 
    lrs = config['lr']

    if config['loss_func'] == 'mse':
        loss_func = F.mse_loss

    for i in range(config['num_inits']):

        if config['calc_cond_num'] == False:
            
            if (config['dataset'] == 'mnist' or config['dataset'] == 'fashion') and config['normalize_mnist'] == True:
                dataset = config['dataset'] + '_' + 'normalized_' + '%d' %i
            else:
                dataset = config['dataset'] + '_' + '%d' %i

            training_information = train_network_configurations(Networks, x_train, y_train, train_dl, loss_func, m_L_config, 
                                                                epochs, lrs, optimizer=config['optimizer'], 
                                                                calc_cond_num=False, calc_every_x_epoch=calc_every_x_epoch,
                                                                verbose_level=config['verbose_level'], seed=config['seed'],device=device, save_model=config['save_model'], save_path=config['save_path'], dataset=dataset)
        else:
            training_information, hessian_information  = train_network_configurations(Networks, x_train, y_train, train_dl, loss_func, m_L_config, 
                                                                                    epochs, lrs, optimizer=config['optimizer'], 
                                                                                    calc_cond_num=True, method_cond_num=config['method_cond_num'], calc_every_x_epoch=calc_every_x_epoch, 
                                                                                    verbose_level=config['verbose_level'], seed=config['seed'],device=device)
            
            
            plot_diff_H_H_O_elementwise_during_training(hessian_information, filetitle=config['experiment_name'], hue_variable=hue_var, size_variable=size_var, file_path=file_path)

            plot_extreme_eigenvalues_during_training(hessian_information, filetitle=config['experiment_name'], hue_variable=hue_var, size_variable=size_var, file_path=file_path)

            plot_Hessian_rank_during_training(hessian_information, filetitle=config['experiment_name'], hue_variable=hue_var, size_variable=size_var, file_path=file_path)

            plot_training_overview_during_training(training_information, hessian_information, title1='MSE of %s networks' %config['activation_func'], filetitle=config['experiment_name'], hue_variable=hue_var, size_variable=size_var, file_path=file_path)
            
        
            time_now = dt.now().isoformat()

            file_path = 'figures/' + config['experiment_name'] + '_' + time_now + '/'
            os.mkdir(file_path)

            yaml_file_name = file_path + 'config_' + time_now + '.yaml'
            with open(yaml_file_name, 'w') as file:
                yaml.dump(config, file)


        

    training_information.to_pickle("pandas_dataframes/training_information_%s.pkl" %config['experiment_name'])

    if config['calc_cond_num'] == True:
        hessian_information.to_pickle("pandas_dataframes/hessian_information_%s.pkl" %config['experiment_name'])


if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)

    plt.rcParams["figure.figsize"] = (4,4)

    # load in YAML configuration
    config = {}
    base_config_path = 'config.yaml'
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

