# Author for computation of Jacobian: ludwigwinkler
# Source: https://discuss.pytorch.org/t/get-gradient-and-jacobian-wrt-the-parameters/98240/6

import future, sys, os, datetime, argparse
from typing import List, Tuple
import numpy as np
import matplotlib

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch, torch.nn
from torch import nn
from torch.nn import Sequential, Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

import copy

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

# from Optimization.BayesianGradients.src.DeterministicLayers import GradBatch_Linear as Linear


def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    '''
        Make params regular Tensors instead of nn.Parameter
    '''
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names

def _set_nested_attr(obj: Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def load_weights(mod: Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)

def compute_jacobian(model, x):
    '''

    @param model: model with vector output (not scalar output!) the parameters of which we want to compute the Jacobian for
    @param x: input since any gradients requires some input
    @return: either store jac directly in parameters or store them differently

    we'll be working on a copy of the model because we don't want to interfere with the optimizers and other functionality
    '''

    jac_model = copy.deepcopy(model) # because we're messing around with parameters (deleting, reinstating etc)
    all_params, all_names = extract_weights(jac_model) # "deparameterize weights"
    load_weights(jac_model, all_names, all_params) # reinstate all weights as plain tensors

    def param_as_input_func(model, x, param):
        load_weights(model, [name], [param]) # name is from the outer scope
        out = model(x)
        return out

    jacobian=np.zeros(1)    
    for i, (name, param) in enumerate(zip(all_names, all_params)):
        jac = torch.autograd.functional.jacobian(lambda param: param_as_input_func(jac_model, x, param), param,
                             strict=True if i==0 else False, vectorize=False if i==0 else True)

        n = jac.shape[0]
        k = jac.shape[1]
        j = torch.reshape(jac,(n,k,jac.shape[-1]*jac.shape[-2]))
#         print(j.shape)
        if i==0:
            jacobian = j
        else:
            jacobian = torch.cat([jacobian,j],dim=2)
#     print(jacobian.shape)        

    del jac_model # cleaning up
    
    return jacobian

def calc_hessian(network,x,y,loss_func):
#     https://stackoverflow.com/questions/74900770/fast-way-to-calculate-hessian-matrix-of-model-parameters-in-pytorch
    '''
    Calculates the full Hessian of the loss function w.r.t. the parameters
    Returns: the Hessian matrix, the spectrum and the matrix rank
    
    network  : Neural network
    x        : input datapoints
    y        : output datapoints/labels
    loss_func: loss function
    '''
    
    # calculate the prediction at initialization (no training)
    y_hat = network(x)
    
    # calculate the MSE given the prediction and the loss
    loss = loss_func(y_hat,y)
    
    num_param = sum(p.numel() for p in network.parameters())

    # Allocate Hessian size
    H = torch.zeros((num_param, num_param))

    # Calculate Jacobian w.r.t. model parameters
    J = torch.autograd.grad(loss, list(network.parameters()), create_graph=True)
    J = torch.cat([e.flatten() for e in J]) # flatten

    # Fill in Hessian
    for i in range(num_param):
        result = torch.autograd.grad(J[i], list(network.parameters()), retain_graph=True)
        H[i] = torch.cat([r.flatten() for r in result]) # flatten
                
    H_spectrum = torch.linalg.eigvalsh(H.detach())
    H_rank = torch.linalg.matrix_rank(H.detach(), atol=1e-7/H.shape[0])
    
    
    return H, H_spectrum, H_rank

def calc_outer_prod_hessian(network,x):
    '''
    Calculates the outer product Hessian of the loss via the Jacobian of the network w.r.t. the parameters
    Returns: the Jacobian times its transposed averaged over all datapoints, and its non-zero spectrum
    
    network: Neural network
    x: input samples
    '''
    
    jacob = compute_jacobian(network,x)
    
    n=jacob.shape[0]
        
    arr = [jacob[i,:,:].T @ jacob[i,:,:] for i in range(n)]
    jac_jac_T = 2*sum(arr)/n
    
#     jac_jac_T_rank = torch.linalg.matrix_rank(jac_jac_T, atol=1e-7/jac_jac_T.shape[0])
    jac_jac_T_rank = torch.linalg.matrix_rank(jac_jac_T)
    
    jac_jac_T_nonzero_spectrum = torch.linalg.eigvalsh(jac_jac_T)[-jac_jac_T_rank:]

    
    return jac_jac_T, jac_jac_T_nonzero_spectrum

# calculate the condition number of the full Hessian and the outer-product Hessian

def calc_condition_num(network,x,y,loss):
    '''
    Calculates the condition number of the full Hessian the outer-product Hessian and the extreme eigenvalues of 
    the full Hessian given a Neural network, the loss function and input & output data
    '''

    H_full, H_spectrum, H_rank = calc_hessian(network,x,y,loss)
    
#     print('H_rank= ', H_rank)
    
    abs_spectrum_sorted = np.sort(np.abs(H_spectrum))
#     print('lam_min= %10.3e' % abs_spectrum_sorted[0])
    
    lam_abs_min_H = abs_spectrum_sorted[-H_rank]
    lam_abs_max_H = abs_spectrum_sorted[-1]

    # append the condition number and the spectrum to list
    H_cond = (abs_spectrum_sorted[-1]/abs_spectrum_sorted[-H_rank])

    # calculate the outer-product Hessian, its nonzero spectrum and its condition number
    H_o, H_o_nonzero_spectrum = calc_outer_prod_hessian(network,x)
    
    lam_abs_min_H_o = np.array(H_o_nonzero_spectrum[0])
    lam_abs_max_H_o = np.array(H_o_nonzero_spectrum[-1])
    
#     print('H_o_rank= ', len(H_o_nonzero_spectrum))

    H_o_cond = np.array(H_o_nonzero_spectrum[-1]/H_o_nonzero_spectrum[0])
    
#     print('H_full', H_full)
#     print('H_o', H_o)
    
    mean_diff_H_H_o = np.array((torch.norm(H_full-H_o)/H_full.shape[0]**2))
    max_diff_H_H_o = np.array(torch.max(H_full-H_o).numpy().astype(float))


    return H_cond, H_o_cond, lam_abs_min_H, lam_abs_max_H, lam_abs_min_H_o, lam_abs_max_H_o, mean_diff_H_H_o, max_diff_H_H_o













