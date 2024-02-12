import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def plot_diff_H_H_O_elementwise_during_training(hessian_information, filetitle='', hue_variable=None, size_variable=None, file_path='figures/', savefig=False):
    '''
    Plot element-wise difference between H and H_O using seaborn

    hessian_information: Pandas dataframe
    filetitle: title of file
    hue_variable: hue parameter in seaborn.relplot
    size_variable: size parameter in seaborn.relplot
    file_path: file path
    savefig: Boolean
    '''

    plt.figure(figsize=(5,5))
    
    sns.set_theme()
    g = sns.relplot(data=hessian_information, kind='line', x='epoch', y= 'mean_diff_H_H_o', 
                    markers=True,
                    style='activ_f',
                    palette='flare', 
                    hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')
    

    plt.title('Average entrywise difference of $H$ and $H_O$ w/ ' + filetitle + ' networks')
    plt.xlabel('Epochs')
    plt.ylabel(r'$\frac{\||H - H_O\||_F}{n^2}$')
        
    filename_1 = file_path + 'mean_diff_H_H_O_elementwise_' + filetitle + '.pdf'

    if savefig:
        plt.savefig(filename_1, bbox_inches='tight')
    
    # plot maximal element-wise difference between H and H_O
    plt.figure(figsize=(5,5))
    
    sns.set_theme()
    g = sns.relplot(data=hessian_information, kind='line', x='epoch', y= 'max_diff_H_H_o', 
                    markers=True,
                    style='activ_f',
                    palette='flare', 
                    hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')
    

    plt.title('Maximal entrywise difference of $H$ and $H_O$ w/ ' + filetitle + ' networks')
    plt.xlabel('Epochs')
    plt.ylabel(r'$\max((H - H_O)_{ij})$')
        
    filename_2 = file_path + 'max_diff_H_H_O_elementwise_' + filetitle + '.pdf'

    if savefig:
        plt.savefig(filename_2, bbox_inches='tight')

    # plot standard deviation of element-wise difference between H and H_O
    plt.figure(figsize=(5,5))
    
    sns.set_theme()
    g = sns.relplot(data=hessian_information, kind='line', x='epoch', y= 'std_diff_H_H_o', 
                    markers=True,
                    style='activ_f',
                    palette='flare', 
                    hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')
    

    plt.title('Standard deviation of entrywise difference of $H$ and $H_O$ w/ ' + filetitle + ' networks')
    plt.xlabel('Epochs')
    plt.ylabel(r'std $((H - H_O)_{ij})$')
        
    filename_3 = file_path + 'std_diff_H_H_O_elementwise_' + filetitle + '.pdf'

    if savefig:
        plt.savefig(filename_3, bbox_inches='tight')

def plot_extreme_eigenvalues_during_training(hessian_information, filetitle='', hue_variable=None, size_variable=None, file_path='figures/', savefig=False):

    '''
    plot the extreme eigenvalues over the course of training

    hessian_information: Pandas dataframe
    filetitle: title of file
    hue_variable: hue parameter in seaborn.relplot
    size_variable: size parameter in seaborn.relplot
    file_path: file path
    savefig: Boolean
    '''
    plt.figure(figsize=(5,5))
    
    sns.set_theme()
    
    g = sns.lineplot(data=hessian_information, x='epoch', y= 'lam_abs_min_H', 
                     markers=True, 
                     style='activ_f',
                     palette='flare',  
                     hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')
    
    plt.xlabel('Epochs')
    plt.ylabel(r'$\lambda_{\min}$')
#     plt.legend(loc=(1,0), ncols=2)
    plt.title('Evolution of smallest eigenvalues during training of ' + filetitle + ' networks')
    
    filename = file_path + 'smallest_evals_during_training_' + filetitle + '.pdf'
    
    if savefig:
        plt.savefig(filename, bbox_inches='tight')


    plt.figure(figsize=(5,5))

    g = sns.lineplot(data=hessian_information, x='epoch', y= 'lam_abs_max_H',
                     markers=True, 
                     style='activ_f', 
                     palette='flare', 
                     hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')

    plt.xlabel('Epochs')
    plt.ylabel(r'$\lambda_{\max}$')
#     plt.legend(loc=(1,0), ncols=2)
    plt.title('Evolution of largest eigenvalues during training of ' + filetitle + ' networks')
    
    filename = file_path + 'largest_evals_during_training_' + filetitle + '.pdf'

    if savefig:
        plt.savefig(filename, bbox_inches='tight')

def plot_Hessian_rank_during_training(hessian_information, filetitle='', hue_variable=None, size_variable=None, file_path='figures/', savefig=False):

    '''
    Plot the evolution of the Hessian rank during training

    hessian_information: Pandas dataframe
    filetitle: title of file
    hue_variable: hue parameter in seaborn.relplot
    size_variable: size parameter in seaborn.relplot
    file_path: file path
    savefig: Boolean
    '''


    plt.figure(figsize=(5,5))
    
    sns.set_theme()
    
    g = sns.lineplot(data=hessian_information, x='epoch', y= 'H_rank', 
                     markers=True, 
                     style='activ_f',
                     palette='flare',  
                     hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')
    
    plt.xlabel('Epochs')
    plt.ylabel(r'rank($H$)')
#     plt.legend(loc=(1,0), ncols=2)
    plt.title('Evolution of full Hessian rank estimate during training of ' + filetitle + ' networks')
    
    if savefig:
        filename = file_path + 'hessian_rank_during_training_' + filetitle + '.pdf'
        plt.savefig(filename, bbox_inches='tight')


    plt.figure(figsize=(5,5))

    g = sns.lineplot(data=hessian_information, x='epoch', y= 'H_o_rank',
                     markers=True, 
                     style='activ_f', 
                     palette='flare', 
                     hue=hue_variable, 
                     size=size_variable)
    g.set(yscale='log')

    plt.xlabel('Epochs')
    plt.ylabel(r'rank($H_O$)')
#     plt.legend(loc=(1,0), ncols=2)
    plt.title('Evolution of outer-product Hessian rank estimate during training of ' + filetitle + ' networks')

    if savefig:
        filename = file_path + 'outer_prod_hessian_rank_during_training_' + filetitle + '.pdf'
        plt.savefig(filename, bbox_inches='tight')

def plot_training_overview_during_training(training_information, hessian_information,
                          title1='MSE of ... Networks', filetitle='',hue_variable=None,size_variable=None, file_path='figures/', savefig=False):
    
#     markers = ['^', 's', 'o', '*', 'p','v', '<']
#     markevery = int(epochs/6)+1
#     markevery2 = int(epochs/(6*calc_every_x_epoch))+1
    
    '''
    Plot training information over course of training network

    training_information: Pandas dataframe
    hessian_information: Pandas dataframe
    title1: Title of figure
    filetitle: file name
    hue_variable: hue parameter in seaborn.relplot
    size_variable: size parameter in seaborn.relplot
    file_path: file path
    savefig: Boolean
    '''

    epsilon = 1e-13

    min_loss_val = min(np.array(training_information['loss']).flatten())
    
    
    
    print('min MSE loss = %1.3e' %min_loss_val)

    loss_to_plot = np.array(training_information['loss']) - min_loss_val + epsilon
    
    training_information = training_information.assign(rel_loss=pd.Series(loss_to_plot.flatten()))

    
    plt.figure(figsize=(10,20))

    plt.subplot(411)
    
    g = sns.lineplot(data=training_information, x='epoch', y='rel_loss', 
                     style='activ_f', palette='flare',
                     hue=hue_variable, 
                     size=size_variable)
    
    g.set(yscale='log')
    
    plt.xlabel('')


    plt.title(title1, fontsize=15)
    plt.ylabel('rel. optimality $f-f_{\min}$', fontsize=15)
#     plt.legend(loc=(1,0), ncols=1, fontsize=15)

    plt.subplot(412)
    
    g = sns.lineplot(data=training_information, x='epoch', y='grad_norm_squared', 
                     style='activ_f', palette='flare',
                     hue=hue_variable, 
                     size=size_variable)

    plt.title('Gradient norm squared', fontsize=15)
    plt.ylabel(r'$\||\nabla f_{\theta} (x) \||^2 $', fontsize=15)
#     plt.legend(loc=(1,0), ncols=1, fontsize=15)

    g.set(yscale='log')
    plt.xlabel('')

    plt.subplot(413)
    
    g = sns.lineplot(data=hessian_information, x='epoch', y='H_cond', 
                     style='activ_f', palette='flare', markers=True,
                     hue=hue_variable, 
                     size=size_variable)
    
    g.set(yscale='log')
    plt.xlabel('')
    
    plt.subplot(414)
    
    g = sns.lineplot(data=hessian_information, x='epoch', y='H_o_cond', 
                     style='activ_f', palette='flare', markers=True,
                     hue=hue_variable, 
                     size=size_variable)
    
    g.set(yscale='log')
    
    plt.xlabel('Epochs')
    
    if savefig:
        filename = file_path + 'training_' + filetitle + '_networks' + '.pdf'

        plt.savefig(filename, bbox_inches='tight')

