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
import seaborn as sns

from torch.distributions.multivariate_normal import MultivariateNormal
from datetime import datetime as dt

from networks import *
from network_derivatives import *
from utils import *

from tqdm import tqdm
import yaml


def train_network(x_train, y_train, train_dl, loss_func, network, optimizer, lr, epochs, calc_cond_num=False, verbose_level=0, calc_every_x_epoch=10):
    
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

    training_information.loc[len(training_information)] = [network.width, network.depth, network.activation_func, 0, loss_func(network(x_train), y_train).detach(), grad_norm_sq]
    
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
                                        'mean_diff_H_H_o':[],
                                        'max_diff_H_H_o':[]
                                        })
 
    if calc_cond_num == True:
        _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _lam_abs_min_H_o, _lam_abs_max_H_o, _mean_diff_H_H_o, _max_diff_H_H_o = calc_condition_num(network,
                                                                    x_train,y_train,
                                                                    loss_func)
        
        hessian_information.loc[len(hessian_information)] = [network.width, network.depth, network.activation_func, 0, _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _mean_diff_H_H_o, _max_diff_H_H_o]
        
    print('Epoch: 0 \t loss= %10.3e' %loss_func(network(x_train), y_train).detach())

    for epoch in tqdm(range(epochs)):
        if calc_cond_num == True and epoch%calc_every_x_epoch==0:
            _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _lam_abs_min_H_o, _lam_abs_max_H_o, _mean_diff_H_H_o, _max_diff_H_H_o = calc_condition_num(network,
                                                                        x_train,y_train,
                                                                        loss_func)

            hessian_information.loc[len(hessian_information)] = [network.width, network.depth, network.activation_func, epoch, _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _mean_diff_H_H_o, _max_diff_H_H_o]            
        
        for xb, yb in train_dl:
            pred = network(xb)        
            loss = loss_func(pred, yb, reduction='mean')

            loss.backward()
            
            grad_norm_sq = sum([torch.linalg.norm(param.grad)**2 for param in network.parameters()])

            training_information.loc[len(training_information)] = [network.width, network.depth, network.activation_func, epoch, loss_func(network(x_train), y_train).detach(), grad_norm_sq]
            
            opt.step()
            opt.zero_grad()

            if verbose_level >= 2:
                print('loss: ', loss_func(network(x_train), y_train).detach())
                    
                print('gradient_norm_sqaured: ', grad_norm_squared)

        
        
        
        if verbose_level == 0 and epoch%int(epochs/10 + 1) ==0:
            print('Epoch: %d \t loss= %10.4e' %(epoch+1, loss_func(network(x_train), y_train).detach()))
        elif verbose_level >= 1:
            print('Epoch: %d \t loss= %10.4e' %(epoch+1, loss_func(network(x_train), y_train).detach()))
         
        
    print('Epoch: %d \t loss= %10.4e' %(epoch+1, loss_func(network(x_train), y_train).detach()))
    if calc_cond_num == True:
        return training_information, hessian_information
    else:
        return training_information

def train_network_configurations(networks, x_train, y_train, train_dl, loss_func, m_L_config, epochs, lrs, optimizer='SGD', calc_cond_num=False, calc_every_x_epoch=10, verbose_level=0, seed=314159):
    '''
    Train list of networks of different "configurations" (#hidden units, #hidden layers)
    networks: List of networks too be trained
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
                                            'mean_diff_H_H_o':[],
                                            'max_diff_H_H_o':[]
                                            })
    


    print('Training Networks...')

    for ind, network in enumerate(networks):

        print('Network configuration: d=%d, k=%d, m=%d, L=%d' % (networks[0].lin_in.weight.shape[1], networks[0].lin_out.weight.shape[0], m_L_config[ind][0], m_L_config[ind][1]+1))

        torch.manual_seed(seed)

        network.init_weights('kaiming_normal')

        if np.isscalar(lrs):
            lr = lrs
        else:
            try:
                lr = lrs[ind]
            except:
                TypeError('Unknown format for learning rate')
        
        if calc_cond_num==False:
            _training_information = train_network(x_train, y_train, 
                                            loss_func, network, 
                                            optimizer=optimizer, 
                                            lr=lr, epochs=epochs, 
                                            calc_cond_num=False, 
                                            verbose_level=verbose_level)
            training_information = pd.concat([training_information, _training_information])
        else: 
            _training_information, _hessian_information = train_network(x_train, y_train, train_dl, 
                                                                        loss_func, network, 
                                                                        optimizer=optimizer, 
                                                                        lr=lr, epochs=epochs, 
                                                                        calc_cond_num=True, 
                                                                        verbose_level=verbose_level,
                                                                        calc_every_x_epoch=calc_every_x_epoch)
            
            training_information = pd.concat([training_information, _training_information])
            hessian_information = pd.concat([hessian_information, _hessian_information])

    training_information['grad_norm_squared'] = training_information['grad_norm_squared'].astype(float)
    training_information['loss'] = training_information['loss'].astype(float)
        
    if calc_cond_num==False:
        
        return training_information,
    else: 
        
        hessian_information['H_o_cond'] = hessian_information['H_o_cond'].astype(float)
        hessian_information['mean_diff_H_H_o'] = hessian_information['mean_diff_H_H_o'].astype(float)
        hessian_information['max_diff_H_H_o'] = hessian_information['max_diff_H_H_o'].astype(float)

        return training_information, hessian_information

def generate_gaussian_data_for_bin_classification(n, mean_1,mean_2,cov_1,cov_2):
    
    # define multi-variate normal generator with given means and covariance
    mvrn_class1 = MultivariateNormal(mean_1,cov_1)
    mvrn_class2 = MultivariateNormal(mean_2,cov_2)
    
    # sample n (input,output) tuples
    x_class1 = mvrn_class1.rsample((n,))    
    x_class2 = mvrn_class2.rsample((n,))
    
    # define labels
    y_class1 = np.ones((n,1))
    y_class2 = -np.ones((n,1))

    x_data = np.concatenate((x_class1,x_class2))
    y_data = np.concatenate((y_class1,y_class2))
    
    # shuffle data
    perm = np.random.permutation(np.arange(2*n))
    x_data = x_data[perm,:]
    y_data = y_data[perm]


    return torch.tensor(x_data), torch.tensor(y_data)

def plot_diff_H_H_O_elementwise(hessian_information, filetitle='', hue_variable=None, size_variable=None):

    # plot average element-wise difference between H and H_O
    plt.figure(figsize=(5,5))
    
    sns.set_theme()
    g = sns.relplot(data=hessian_information, kind='line', x='epoch', y= 'mean_diff_H_H_o', 
                    markers=True,
                    style='activ_f',
                    palette='colorblind', 
                    hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')
    

    plt.title('Average entrywise difference of $H$ and $H_O$ w/ ' + filetitle + ' networks')
    plt.xlabel('Epochs')
    plt.ylabel(r'$\frac{\||H - H_O\||_F}{n^2}$')
        
    filename_1 = 'figures/' + 'mean_diff_H_H_O_elementwise_' + filetitle + '.pdf'

    plt.savefig(filename_1, bbox_inches='tight')
    
    # plot maximal element-wise difference between H and H_O
    plt.figure(figsize=(5,5))
    
    sns.set_theme()
    g = sns.relplot(data=hessian_information, kind='line', x='epoch', y= 'max_diff_H_H_o', 
                    markers=True,
                    style='activ_f',
                    palette='colorblind', 
                    hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')
    

    plt.title('Maximal entrywise difference of $H$ and $H_O$ w/ ' + filetitle + ' networks')
    plt.xlabel('Epochs')
    plt.ylabel(r'$\max((H - H_O)_{ij})$')
        
    filename_2 = 'figures/' + 'max_diff_H_H_O_elementwise_' + filetitle + '.pdf'

    plt.savefig(filename_2, bbox_inches='tight')

def plot_extreme_eigenvalues(hessian_information, filetitle='', hue_variable=None, size_variable=None):

    plt.figure(figsize=(5,5))
    
    sns.set_theme()
    
    g = sns.lineplot(data=hessian_information, x='epoch', y= 'lam_abs_min_H', 
                     markers=True, 
                     style='activ_f', 
                     palette='colorblind', 
                     hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')
    
    plt.xlabel('Epochs')
    plt.ylabel(r'$\lambda_{\min}$')
#     plt.legend(loc=(1,0), ncols=2)
    plt.title('Evolution of smallest eigenvalues during training of ' + filetitle + ' networks')
    
    filename = 'figures/' + 'smallest_evals_during_training_' + filetitle + '.pdf'

    plt.savefig(filename, bbox_inches='tight')


    plt.figure(figsize=(5,5))

    g = sns.lineplot(data=hessian_information, x='epoch', y= 'lam_abs_max_H',
                     markers=True, 
                     style='activ_f', 
                     palette='colorblind', 
                     hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')

    plt.xlabel('Epochs')
    plt.ylabel(r'$\lambda_{\max}$')
#     plt.legend(loc=(1,0), ncols=2)
    plt.title('Evolution of largest eigenvalues during training of ' + filetitle + ' networks')
    
    filename = 'figures/' + 'largest_evals_during_training_' + filetitle + '.pdf'

    plt.savefig(filename, bbox_inches='tight')

def plot_training_overview(training_information, hessian_information,
                          title1='MSE of ... Networks', filetitle='',hue_variable=None,size_variable=None):
    
#     markers = ['^', 's', 'o', '*', 'p','v', '<']
#     markevery = int(epochs/6)+1
#     markevery2 = int(epochs/(6*calc_every_x_epoch))+1

    epsilon = 1e-13

    min_loss_val = min(np.array(training_information['loss']).flatten())
    
    
    
    print('min MSE loss = %1.3e' %min_loss_val)

    loss_to_plot = np.array(training_information['loss']) - min_loss_val + epsilon
    
    training_information = training_information.assign(rel_loss=pd.Series(loss_to_plot.flatten()))

    
    plt.figure(figsize=(10,20))

    plt.subplot(411)
    
    g = sns.lineplot(data=training_information, x='epoch', y='rel_loss', 
                     style='activ_f', palette='colorblind',
                     hue=hue_variable, 
                     size=size_variable)
    
    g.set(yscale='log')
    
    plt.xlabel('')


    plt.title(title1, fontsize=15)
    plt.ylabel('rel. optimality $f-f_{\min}$', fontsize=15)
#     plt.legend(loc=(1,0), ncols=1, fontsize=15)

    plt.subplot(412)
    
    g = sns.lineplot(data=training_information, x='epoch', y='grad_norm_squared', 
                     style='activ_f', palette='colorblind',
                     hue=hue_variable, 
                     size=size_variable)

    plt.title('Gradient norm squared', fontsize=15)
    plt.ylabel(r'$\||\nabla f_{\theta} (x) \||^2 $', fontsize=15)
#     plt.legend(loc=(1,0), ncols=1, fontsize=15)

    g.set(yscale='log')
    plt.xlabel('')

    plt.subplot(413)
    
    g = sns.lineplot(data=hessian_information, x='epoch', y='H_cond', 
                     style='activ_f', palette='colorblind', markers=True,
                     hue=hue_variable, 
                     size=size_variable)
    
    g.set(yscale='log')
    plt.xlabel('')
    
    plt.subplot(414)
    
    g = sns.lineplot(data=hessian_information, x='epoch', y='H_o_cond', 
                     style='activ_f', palette='colorblind', markers=True,
                     hue=hue_variable, 
                     size=size_variable)
    
    g.set(yscale='log')
    
    plt.xlabel('Epochs')
    
    filename = 'figures/' + 'training_' + filetitle + '_networks' + '.pdf'

    plt.savefig(filename, bbox_inches='tight')


def main(project_name, experiment_name, config):

    print('Running experiment %s for project %s ' %(experiment_name, project_name))

    #TODO: Create new folder to save generated figures in this folder 
    #TODO: with specifications from config file
    #TODO: timestamp

    # setting up Gaussian dataset

    mean_1 = -torch.tensor(config['mean_class1'])
    mean_2 = torch.tensor(config['mean_class2'])

    cov_1 = torch.tensor(config['cov_class1'])
    cov_2 = torch.tensor(config['cov_class2'])

    torch.manual_seed(config['seed'])

    n = config['data_pts_per_class_training']
    m = config['data_pts_per_class_validation']
    v = config['data_pts_per_class_testing']

    x_train, y_train = generate_gaussian_data_for_bin_classification(n, mean_1,mean_2,cov_1,cov_2)
    x_test, y_test = generate_gaussian_data_for_bin_classification(m, mean_1,mean_2,cov_1,cov_2)
    x_val, y_val = generate_gaussian_data_for_bin_classification(v, mean_1,mean_2,cov_1,cov_2)

    train_dl, valid_dl, test_dl = create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, config['batch_size'])

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
    k = 1 # output dimension
    activ_func = config['activation_func']

    L = np.array(config['hidden_layers'])-1 # number of hidden layers of dim "m"       
        
    m_L_config = [] # keep track of the network configuration
    num_param = [] # count the number of parameters in the model

    Networks = [] # list of NN with different configurations


    # initiate networks of given depth L[l] with m1 hidden units each
    for m in m1:
        for l in L:
            m_L_config.append((m,l))
            if config['model_name'] == 'sequential':

                if config['activation_func'] == 'leaky_relu':
                    kwargs={'neg_slope': config['neg_slope']} # negative slope of leaky ReLU
                    Networks.append(Sequential_NN(d,m,k,l,'leaky_relu',**kwargs))
                else:
                    Networks.append(Sequential_NN(d,m,k,l,activ_func))
            elif config['model_name'] == 'sequential_w_fully_skip':
                if config['activation_func'] == 'leaky_relu':
                    kwargs={'neg_slope': config['neg_slope']} # negative slope of leaky ReLU
                    Networks.append(Sequential_fully_skip_NN(d,m,k,l,beta=config['beta'],
                                                            activation=activ_func,**kwargs))
                else: 
                    Networks.append(Sequential_fully_skip_NN(d,m,k,l,beta=config['beta'],
                                                            activation='linear'))
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

    if config['calc_cond_num'] == False:
        training_information = train_network_configurations(Networks, x_train, y_train, train_dl, loss_func, m_L_config, 
                                                            epochs, lrs, optimizer=config['optimizer'], 
                                                            calc_cond_num=False, 
                                                            verbose_level=config['config_level'], seed=config['seed'])
    else:
        training_information, hessian_information  = train_network_configurations(Networks, x_train, y_train, train_dl, loss_func, m_L_config, 
                                                                                epochs, lrs, optimizer=config['optimizer'], 
                                                                                calc_cond_num=True, calc_every_x_epoch=calc_every_x_epoch, 
                                                                                verbose_level=config['verbose_level'], seed=config['seed'])
        
        plot_diff_H_H_O_elementwise(hessian_information, filetitle=config['experiment_name'], hue_variable='width', size_variable='depth')

        plot_extreme_eigenvalues(hessian_information, filetitle=config['experiment_name'], hue_variable='width', size_variable='depth')

        plot_training_overview(training_information, hessian_information, title1='MSE of %s networks' %config['activation_func'], filetitle=config['experiment_name'], hue_variable='width', size_variable='depth')

    training_information.to_pickle("pandas_dataframes/training_information_%s.pkl" %config['experiment_name'])
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

