import torch
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from pathlib import Path
import requests
import pickle
import gzip

import mnist_reader

def load_mnist(dataset, n, downsample_factor, whiten, device):
    '''
    load MNIST or Fashion MNIST dataset

    dataset: mnist or fashion
    n: number of samples
    downsample_factor: factor by which the image is downsampled. For unchanged data use factor of 1
    whiten: Boolean: whiten data, such that the covariance matrix is identity
    device: device on which data is to be loaded: cpu or cuda
    
    Note that data is also centered
    '''
    
    if dataset == 'mnist':
        DATA_PATH = Path("data")
        PATH = DATA_PATH / "mnist"

        PATH.mkdir(parents=True, exist_ok=True)

        URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
        FILENAME = "mnist.pkl.gz"

        if not (PATH / FILENAME).exists():
            content = requests.get(URL + FILENAME).content
            (PATH / FILENAME).open("wb").write(content)
            
        with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
            
        print(y_train.dtype)
    elif dataset == 'fashion':
        x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
        x_valid, y_valid = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # subsample data
    y_train = y_train[0:n]
    x_train = x_train[0:n]
    
    # downsample input data
    dim = int(np.ceil(28/downsample_factor))

    print('Downsampling MNIST data by a factor of %d' %downsample_factor)
    print('Resulting image size: (%d x %d) ' %(dim,dim))
    print('Subsampling %d data points.' %n)

    x_train_downsampled = torch.zeros((x_train.shape[0],dim*dim))
    x_valid_downsampled = torch.zeros((x_valid.shape[0],dim*dim))

    for i in range(x_train.shape[0]):
        tmp = x_train[i].reshape((28, 28))
        tmp_downsampled = tmp[0::downsample_factor,0::downsample_factor]
        x_train_downsampled[i] = torch.tensor(tmp_downsampled.flatten())
        
    # x_train_downsampled = torch.tensor(x_train_downsampled).to(device)
        
    for i in range(x_valid.shape[0]):
        tmp = x_valid[i].reshape((28, 28))
        tmp_downsampled = tmp[0::downsample_factor,0::downsample_factor]
        x_valid_downsampled[i] = torch.tensor(tmp_downsampled.flatten())

    if whiten==True:
        
        U, s, Vt = np.linalg.svd(x_train_downsampled, full_matrices=False)

        # U and Vt are the singular matrices, and s contains the singular values.
        # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
        # will be white
        x_train_downsampled = np.dot(U, Vt)
        
        U, s, Vt = np.linalg.svd(x_valid_downsampled, full_matrices=False)
        x_valid_downsampled = np.dot(U, Vt)
       
#     x_train_downsampled = x_train_downsampled - np.mean(x_train_downsampled, axis=0)
#     x_valid_downsampled = x_valid_downsampled - np.mean(x_valid_downsampled, axis=0)
    

    # convert labels into 1-hot encoding
    y_train = torch.tensor(y_train, dtype=torch.int64, device=device)
    y_train_onehot = F.one_hot(y_train)
    y_train_onehot = y_train_onehot.to(torch.float64)

    y_valid = torch.tensor(y_valid, dtype=torch.int64, device=device)
    y_valid_onehot = F.one_hot(y_valid)
    y_valid_onehot = y_valid_onehot.to(torch.float64)

    
        
    # x_valid_downsampled = torch.tensor(x_valid_downsampled).to(device)


    return torch.tensor(x_train_downsampled).to(device), y_train_onehot, torch.tensor(x_valid_downsampled).to(device), y_valid_onehot


def load_cifar10(n, grayscale, flatten, whiten, device):
    '''
    load Cifar-10 dataset

    n: number of samples
    grayscale: Boolean, whether grayscale Cifar-10 data should be loaded or not
    flatten: Boolean: input images as flattened vectors
    whiten: Boolean: preprocess input, by whitening the data
    device: cpu or cuda
    '''
    

    if grayscale == True:
        with open('data/cifar10/cifar10_grayscale.pkl', 'rb') as file:
            (x_train, y_train, x_valid, y_valid) = pickle.load(file)
        file.close()
    else:
        with open('data/cifar10/cifar10.pkl', 'rb') as file:
            (x_train, y_train, x_valid, y_valid) = pickle.load(file)
        file.close()
        
    if flatten == True:
        x_train = torch.reshape(x_train,[x_train.shape[0],-1])
        x_valid = torch.reshape(x_train,[x_valid.shape[0],-1])
    
    # subsample data
    x_train = x_train[:n,:]
    x_valid = x_valid[:n,:]
    
    if whiten==True:
        
        U, s, Vt = np.linalg.svd(x_train, full_matrices=False)

        # U and Vt are the singular matrices, and s contains the singular values.
        # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
        # will be white
        x_train = np.dot(U, Vt)
        
        U, s, Vt = np.linalg.svd(x_valid, full_matrices=False)
        x_valid = np.dot(U, Vt)
    
    # convert labels into 1-hot encoding
    y_train = torch.tensor(y_train, dtype=torch.int64, device=device)
    y_train = y_train[:n]
    
    y_train_onehot = F.one_hot(y_train)
    y_train_onehot = y_train_onehot.to(torch.float64)

    y_valid = torch.tensor(y_valid, dtype=torch.int64, device=device)
    y_valid = y_valid[:n]
    
    y_valid_onehot = F.one_hot(y_valid)
    y_valid_onehot = y_valid_onehot.to(torch.float64)
    
    print('Loaded Cifar-10...')
    print('x_train.shape: ', x_train.shape)
    
    return torch.tensor(x_train).to(device), y_train_onehot.to(device), torch.tensor(x_valid).to(device), y_valid_onehot.to(device)


def generate_gaussian_data_for_bin_classification(n, d, k, whiten, seed, device):
    '''
    Generate bimodal Gaussian data for classification
    In the current implementation, the mean of both clusters is set to +/- the all-ones vector 
    and the covariance of both clusters is the identity matrix. 
    The labels are also +/- the all-ones vector. However in the classification setting only k=1 makes really sense...

    n: number of samples
    seed: seed for random number generator
    d: input dimension (dimension of the clusters)
    k: output dimension (number of classes, which is 2)
    seed:
    '''
    
    # rng = torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    

    mean_1 = torch.ones((d,))
    mean_2 = -torch.ones((d,))

    cov_1 = torch.eye(d)
    cov_2 = torch.eye(d)

    # define multi-variate normal generator with given means and covariance
    mvrn_class1 = MultivariateNormal(mean_1, cov_1)
    mvrn_class2 = MultivariateNormal(mean_2, cov_2)
    
    # sample n (input,output) tuples
    x_class1 = mvrn_class1.rsample((n,))    
    x_class2 = mvrn_class2.rsample((n,))
    
    # define labels
    y_class1 = torch.ones((n,k),device=device)
    y_class2 = -torch.ones((n,k),device=device)

    x_data = torch.concatenate((x_class1,x_class2))
    y_data = torch.concatenate((y_class1,y_class2))
    
    # shuffle data
    perm = np.random.permutation(torch.arange(2*n))
    x_data = x_data[perm,:]
    y_data = y_data[perm]
    
    if whiten==True:
        
        U, s, Vt = np.linalg.svd(x_data, full_matrices=False)

        # U and Vt are the singular matrices, and s contains the singular values.
        # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
        # will be white
        x_data = np.dot(U, Vt)


    return torch.tensor(x_data).to(device), torch.tensor(y_data).to(device)

