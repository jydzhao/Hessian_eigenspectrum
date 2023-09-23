from torch import nn
import numpy as np
import torch


# Create model of Sequential NN with L hidden layers and fixed hidden width m
# input dim = d, hidden dim = m, output dim = k
class Sequential_NN(nn.Module):
    def __init__(self,d,m,k,L,activation='linear', **kwargs):
        """
            d: input dimension
            m: hidden layer dimension 
            k: output dimension
            L: number of hidden layers
        """
        super().__init__()
        
        self.activation_func = activation
        
        if activation=='linear':
            self.activ = nn.Identity()
        elif activation=='relu':
            self.activ = nn.ReLU()
        elif activation=='leaky_relu':
            self.activ = nn.LeakyReLU(negative_slope = kwargs['neg_slope'])
        elif activation=='gelu':
            self.activ = nn.GELU()
        
        self.L = L
        
        self.lin_out = nn.Linear(m, k, bias=True)
        self.lin_in = nn.Linear(d, m, bias=True)
        
        self.lin_hidden = nn.ModuleList([nn.Linear(m, m, bias=True) for i in range(self.L)])
        
        self.sequential = nn.Sequential(self.lin_in)
        
        for i in range(self.L):
            self.sequential.append(self.activ)
            self.sequential.append(self.lin_hidden[i])
        
        self.sequential.append(self.activ)
        self.sequential.append(self.lin_out) 
        
        
    def forward(self, xb):
        xb = self.sequential(xb)
        
        return xb
    
    def init_weights(self, init_type, *kwargs):

        if init_type == 'kaiming_normal':
            try:
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity=self.activation_func)
                torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity=self.activation_func)
                for i in range(self.L):
                    torch.nn.init.normal_(self.lin_hidden[i].bias)
                    torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity=self.activation_func)
            except:
                print('Unsupported activation function. Using "linear" as non-linearity argument')
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
                torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
                for i in range(self.L):
                    torch.nn.init.normal_(self.lin_hidden[i].bias) 
                    torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')       
        elif init_type == 'xavier_normal':
            torch.nn.init.normal_(self.lin_in.bias)
            torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.xavier_normal_(self.lin_in.weight)
            torch.nn.init.xavier_normal_(self.lin_out.weight)
            for i in range(self.L):
                torch.nn.init.normal_(self.lin_hidden[i].bias)
                torch.nn.init.xavier_normal_(self.lin_hidden[i].weight)
        elif init_type == 'all_const':
            torch.nn.init.constant_(self.lin_in.bias, kwargs[0])
            torch.nn.init.constant_(self.lin_out.bias, kwargs[0])
            torch.nn.init.constant_(self.lin_in.weight, kwargs[0])
            torch.nn.init.constant_(self.lin_out.weight, kwargs[0])
            for i in range(self.L):
                torch.nn.init.constant_(self.lin_hidden[i].bias, kwargs[0])
                torch.nn.init.constant_(self.lin_hidden[i].weight, kwargs[0])
        else:
            print('Unknown initialization. Using Kaiming normal initialization with linear as nonlinearity argument')
            torch.nn.init.normal_(self.lin_in.bias)
            torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
            torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
            for i in range(self.L):
                torch.nn.init.normal_(self.lin_hidden[i].bias) 
                torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')                        

      
# Create sequential model with fully connected skip connections
# input dim = d, hidden dim = m, output dim = k
class Sequential_fully_skip_NN(nn.Module):
    def __init__(self,d,m,k,L,beta=1,activation='linear', **kwargs):
        """
            d: input dimension
            m: hidden layer dimension 
            k: output dimension
            L: number of hidden layers
            beta: scale of residual connections
        """
        super().__init__()
        
        self.activation_func = activation
        
        if activation=='linear':
            self.activ = nn.Identity()
        elif activation=='relu':
            self.activ = nn.ReLU()
        elif activation=='leaky_relu':
            self.activ = nn.LeakyReLU(negative_slope = kwargs['neg_slope'])
        elif activation=='gelu':
            self.activ = nn.GELU()
        
        self.beta = beta
        self.L = L
        self.lin_out = nn.Linear(m, k, bias=True)
        self.lin_in = nn.Linear(d, m, bias=True)
        
        self.lin_hidden = nn.ModuleList([nn.Linear(m, m, bias=True) for i in range(self.L)])
        
        
    def forward(self, xb):
        
        xbs = [xb]
        
        xb = self.lin_in(xb)
        for x in xbs:
            xb +=  x @ (self.beta * torch.eye(x.shape[1],xb.shape[1]))
        
        xb = self.activ(xb)
        xbs.append(xb)
        
        for i in range(self.L):
            xb = self.lin_hidden[i](xb)
            
            for x in xbs:
                xb += x @ (self.beta * torch.eye(x.shape[1],xb.shape[1]))
                
            xb = self.activ(xb)
            xbs.append(xb)
            
        xb = self.lin_out(xb)
        
        return xb
    
    def init_weights(self, init_type, *kwargs):

        if init_type == 'kaiming_normal':
            try:
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity=self.activation_func)
                torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity=self.activation_func)
                for i in range(self.L):
                    torch.nn.init.normal_(self.lin_hidden[i].bias)
                    torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity=self.activation_func)
            except:
                print('Unsupported activation function. Using "linear" as non-linearity argument')
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
                torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
                for i in range(self.L):
                    torch.nn.init.normal_(self.lin_hidden[i].bias) 
                    torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')       
        elif init_type == 'xavier_normal':
            torch.nn.init.normal_(self.lin_in.bias)
            torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.xavier_normal_(self.lin_in.weight)
            torch.nn.init.xavier_normal_(self.lin_out.weight)
            for i in range(self.L):
                torch.nn.init.normal_(self.lin_hidden[i].bias)
                torch.nn.init.xavier_normal_(self.lin_hidden[i].weight)
        elif init_type == 'all_const':
            torch.nn.init.constant_(self.lin_in.bias, kwargs[0])
            torch.nn.init.constant_(self.lin_out.bias, kwargs[0])
            torch.nn.init.constant_(self.lin_in.weight, kwargs[0])
            torch.nn.init.constant_(self.lin_out.weight, kwargs[0])
            for i in range(self.L):
                torch.nn.init.constant_(self.lin_hidden[i].bias, kwargs[0])
                torch.nn.init.constant_(self.lin_hidden[i].weight, kwargs[0])
        else:
            print('Unknown initialization. Using Kaiming normal initialization with linear as nonlinearity argument')
            torch.nn.init.normal_(self.lin_in.bias)
            torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
            torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
            for i in range(self.L):
                torch.nn.init.normal_(self.lin_hidden[i].bias) 
                torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')              

# Create Sequential model with L hidden layers and varying number of hidden units
class Sequential_NN_varying_hidden_units(nn.Module):
    def __init__(self,d,m,k,L,activation='linear', **kwargs):
        """
            d: input dimension
            m: hidden layer dimension 
            k: output dimension
            L: number of hidden layers
        """
        super().__init__()
        
        self.activation_func = activation
        
        if activation=='linear':
            self.activ = nn.Identity()
        elif activation=='relu':
            self.activ = nn.ReLU()
        elif activation=='leaky_relu':
            self.activ = nn.LeakyReLU(negative_slope = kwargs['neg_slope'])
        elif activation=='gelu':
            self.activ = nn.GELU()
        
        self.L = L
        self.lin_out = nn.Linear(m[-1], k, bias=False)
        self.lin_in = nn.Linear(d, m[0], bias=False)
        
        self.lin_hidden = nn.ModuleList()
        for i in range(self.L):
            self.lin_hidden.append(nn.Linear(m[i], m[i+1], bias=False))

        self.sequential = nn.Sequential(self.lin_in)
        
        for i in range(self.L):
            self.sequential.append(self.activ)
            self.sequential.append(self.lin_hidden[i])
        
        self.sequential.append(self.activ)
        self.sequential.append(self.lin_out)                
        
        
    def forward(self, xb):
        xb = self.sequential(xb)       
        return xb
    
    def init_weights(self, init_type, *kwargs):
        if init_type == 'kaiming_normal':
            try:
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity=self.activation_func)
                torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity=self.activation_func)
                for i in range(self.L):
                    torch.nn.init.normal_(self.lin_hidden[i].bias)
                    torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity=self.activation_func)
            except:
                print('Unsupported activation function. Using "linear" as non-linearity argument')
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
                torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
                for i in range(self.L):
                    torch.nn.init.normal_(self.lin_hidden[i].bias) 
                    torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')       
        elif init_type == 'xavier_normal':
            torch.nn.init.normal_(self.lin_in.bias)
            torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.xavier_normal_(self.lin_in.weight)
            torch.nn.init.xavier_normal_(self.lin_out.weight)
            for i in range(self.L):
                torch.nn.init.normal_(self.lin_hidden[i].bias)
                torch.nn.init.xavier_normal_(self.lin_hidden[i].weight)
        elif init_type == 'all_const':
            torch.nn.init.constant_(self.lin_in.bias, kwargs[0])
            torch.nn.init.constant_(self.lin_out.bias, kwargs[0])
            torch.nn.init.constant_(self.lin_in.weight, kwargs[0])
            torch.nn.init.constant_(self.lin_out.weight, kwargs[0])
            for i in range(self.L):
                torch.nn.init.constant_(self.lin_hidden[i].bias, kwargs[0])
                torch.nn.init.constant_(self.lin_hidden[i].weight, kwargs[0])
        else:
            print('Unknown initialization. Using Kaiming normal initialization with linear as nonlinearity argument')
            torch.nn.init.normal_(self.lin_in.bias)
            torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
            torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
            for i in range(self.L):
                torch.nn.init.normal_(self.lin_hidden[i].bias) 
                torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')                        

# Create model of linear NN with L hidden layers
# input dim = d, hidden dim = m, output dim = k
class Linear_skip_single_layer_NN(nn.Module):
    def __init__(self,d,m,k,L,beta=1):
        """
            d: input dimension
            m: hidden layer dimension 
            k: output dimension
            L: number of hidden layers
            beta: scale of residual connections
        """
        super().__init__()
        
        self.beta = beta
        self.L = L
 
        self.lin_out = nn.Linear(m, k, bias=False)
        self.lin_in = nn.Linear(d, m, bias=False)
        
        self.lin_hidden = nn.ModuleList([nn.Linear(m, m, bias=False) for i in range(self.L)])
        
        
    def forward(self, xb):
        
        xb = self.lin_in(xb) + xb @ (self.beta * torch.eye(self.lin_in.weight.shape[1],self.lin_in.weight.shape[0]))
                
        for i in range(self.L):
            xb = self.lin_hidden[i](xb) + xb @ (self.beta * torch.eye(self.lin_hidden[i].weight.shape[1],self.lin_hidden[i].weight.shape[0])) 
            
            
        xb = self.lin_out(xb) + xb @ (self.beta * torch.eye(self.lin_out.weight.shape[1],self.lin_out.weight.shape[0]))
        
        return xb
    
    def init_weights(self, init_type, *kwargs):
        if init_type == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
            torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
            for i in range(self.L):
                torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')
        elif init_type == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.lin_in.weight, nonlinearity='linear')
            torch.nn.init.kaiming_uniform_(self.lin_out.weight, nonlinearity='linear')
            for i in range(self.L):
                torch.nn.init.kaiming_uniform_(self.lin_hidden[i].weight, nonlinearity='linear')
        elif init_type == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.lin_in.weight)
            torch.nn.init.xavier_normal_(self.lin_out.weight)
            for i in range(self.L):
                torch.nn.init.xavier_normal_(self.lin_hidden[i].weight)
        elif init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.lin_in.weight, nonlinearity='linear')
            torch.nn.init.xavier_uniform_(self.lin_out.weight, nonlinearity='linear')
            for i in range(self.L):
                torch.nn.init.xavier_uniform_(self.lin_hidden[i].weight, nonlinearity='linear')
        elif init_type == 'all_const':
            torch.nn.init.constant_(self.lin_in.weight, kwargs[0])
            torch.nn.init.constant_(self.lin_out.weight, kwargs[0])
            for i in range(self.L):
                torch.nn.init.constant_(self.lin_hidden[i].weight, kwargs[0])
        else:
            print('Unknown initialization. Using Kaiming normal initialization')
            torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
            torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
            for i in range(self.L):
                torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')             
                