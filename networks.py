from torch import nn
import numpy as np
import torch


# Create model of Sequential NN with L hidden layers and fixed hidden width m
# input dim = d, hidden dim = m, output dim = k
class Sequential_NN(nn.Module):
    def __init__(self,d,m,k,L,bias,activation='linear', **kwargs):
        """
            d: input dimension
            m: hidden layer dimension 
            k: output dimension
            L: number of hidden layers
        """
        super().__init__()
        
        self.input_dim = d
        self.output_dim = k
        self.width = m
        self.depth = L+1
        self.activation_func = activation
        self.bias = bias
        print('bias',bias)
        
        if activation=='linear':
            self.activ = nn.Identity()
        elif activation=='relu':
            self.activ = nn.ReLU()
        elif activation=='leaky_relu':
            self.activ = nn.LeakyReLU(negative_slope = kwargs['neg_slope'])
        elif activation=='gelu':
            self.activ = nn.GELU()
        
        self.L = L
        
        self.lin_out = nn.Linear(m, k, bias=False)
        self.lin_in = nn.Linear(d, m, bias=False)
        
        self.lin_hidden = nn.ModuleList([nn.Linear(m, m, bias=False) for i in range(self.L)])
        
        
        if kwargs['batch_norm'] == True:
            print('Using batch norm after all layers')
            self.batch_norm = True
            self.bn1 = torch.nn.BatchNorm1d( d )
            self.bn2 = torch.nn.BatchNorm1d( m )
        else:
            self.batch_norm = False
        
        self.sequential = nn.Sequential(self.lin_in)

        
            
        for i in range(self.L):
            self.sequential.append(self.activ)
            if self.batch_norm:
                if i == 0:
                    self.sequential.append(self.bn1)
                else: 
                    self.sequential.append(self.bn2)
                    
            self.sequential.append(self.lin_hidden[i])
            
        
        self.sequential.append(self.activ)
        
        if self.batch_norm:
            self.sequential.append(self.bn2)
            
        self.sequential.append(self.lin_out) 
        
    def forward(self, xb):
        xb = self.sequential(xb)
        
        return xb
    
    def init_weights(self, init_type, *kwargs):

        if init_type == 'kaiming_normal':
            try:
                if self.bias:
                    torch.nn.init.normal_(self.lin_in.bias)
                    torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity=self.activation_func)
                torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity=self.activation_func)
                for i in range(self.L):
                    if self.bias:
                        torch.nn.init.normal_(self.lin_hidden[i].bias)
                    torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity=self.activation_func)
            except:
                print('Unsupported activation function. Using "linear" as non-linearity argument')
                if self.bias:
                    torch.nn.init.normal_(self.lin_in.bias)
                    torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
                torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
                for i in range(self.L):
                    if self.bias:
                        torch.nn.init.normal_(self.lin_hidden[i].bias) 
                    torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')       
        elif init_type == 'xavier_normal':
            if self.bias:
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.xavier_normal_(self.lin_in.weight)
            torch.nn.init.xavier_normal_(self.lin_out.weight)
            for i in range(self.L):
                if self.bias:
                    torch.nn.init.normal_(self.lin_hidden[i].bias)
                torch.nn.init.xavier_normal_(self.lin_hidden[i].weight)
        elif init_type == 'all_const':
            if self.bias:
                torch.nn.init.constant_(self.lin_in.bias, kwargs[0])
                torch.nn.init.constant_(self.lin_out.bias, kwargs[0])
            torch.nn.init.constant_(self.lin_in.weight, kwargs[0])
            torch.nn.init.constant_(self.lin_out.weight, kwargs[0])
            for i in range(self.L):
                if self.bias:
                    torch.nn.init.constant_(self.lin_hidden[i].bias, kwargs[0])
                torch.nn.init.constant_(self.lin_hidden[i].weight, kwargs[0])
        elif init_type == 'orthogonal':
            if self.bias:
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.orthogonal_(self.lin_in.weight)
            torch.nn.init.orthogonal_(self.lin_out.weight)

            for i in range(self.L):
                if self.bias:
                    torch.nn.init.normal_(self.lin_hidden[i].bias)
                torch.nn.init.orthogonal_(self.lin_hidden[i].weight)
        else:
            print('Unknown initialization. Using Kaiming normal initialization with linear as nonlinearity argument')
            if self.bias:
                torch.nn.init.normal_(self.lin_in.bias)
                torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
            torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
            for i in range(self.L):
                if self.bias:
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
            activation: activation function: linear, relu, leaky_relu, gelu
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
        
        self.input_dim = d
        self.output_dim = k
        self.beta = beta
        self.L = L
        self.depth = L+1
        self.width = m
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
        
        self.activation_func = 'linear'
        self.input_dim = d
        self.output_dim = k
        self.beta = beta
        self.L = L
        self.depth = L+1
        self.width = m
        
 
        self.lin_out = nn.Linear(m, k, bias=False)
        self.lin_in = nn.Linear(d, m, bias=False)
        
        self.lin_hidden = nn.ModuleList([nn.Linear(m, m, bias=False) for i in range(self.L)])
        
        
    def forward(self, xb):
        
        device = xb.device
        
        xb = self.lin_in(xb) + xb @ (self.beta * torch.eye(self.lin_in.weight.shape[1],self.lin_in.weight.shape[0]).to(device))
                
        for i in range(self.L):
            xb = self.lin_hidden[i](xb) + xb @ (self.beta * torch.eye(self.lin_hidden[i].weight.shape[1],self.lin_hidden[i].weight.shape[0])).to(device)
            
            
        xb = self.lin_out(xb) + xb @ (self.beta * torch.eye(self.lin_out.weight.shape[1],self.lin_out.weight.shape[0])).to(device)
        
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

# Lambda Class to preprocess image for MNIST dataset
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)

# Create model of Sequential convolutional NN with L hidden layers and fixed hidden width m
# input dim = d, hidden dim = m, output dim = k
class Sequential_CNN(nn.Module):
    def __init__(self,m,L, kernel_size=3, activation='linear', **kwargs):
        """
           
            m: hidden layer dimension 
            
            L: number of hidden layers
        """
        super().__init__()
        
#         self.input_dim = d
#         self.output_dim = k
        self.num_channels = m
        self.depth = L
        self.activation_func = activation
        self.kernel_size = kernel_size
        
        if activation=='linear':
            self.activ = nn.Identity()
        elif activation=='relu':
            self.activ = nn.ReLU()
        elif activation=='leaky_relu':
            self.activ = nn.LeakyReLU(negative_slope = kwargs['neg_slope'])
        elif activation=='gelu':
            self.activ = nn.GELU()
        
        self.L = L
        
#         self.conv_out = nn.Linear(m, k, kernel_size=kernel_size,bias=False)
        self.conv_in = nn.Conv1d(1, m, kernel_size=kernel_size, bias=False)
        
        self.conv_hidden = nn.ModuleList([nn.Conv1d(m, m, kernel_size=kernel_size, bias=False) for i in range(self.L)])

        
        self.sequential = nn.Sequential(self.conv_in)

        
            
        for i in range(self.L):
            self.sequential.append(self.activ)
                    
            self.sequential.append(self.conv_hidden[i])
            
        
#         self.sequential.append(self.activ)
            
#         self.sequential.append(self.conv_out)
        
#         self.sequential.append(Lambda(lambda x: x.view(x.size(1), -1)))
                               
        
    def forward(self, xb):
        xb = self.sequential(xb)
#         print(xb.shape)
#         xb = xb.view(-1, xb.size(1))
        return xb
    
    def init_weights(self, init_type, *kwargs):

        if init_type == 'kaiming_normal':
            try:
#                 torch.nn.init.normal_(self.lin_in.bias)
#                 torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity=self.activation_func)
#                 torch.nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity=self.activation_func)
                for i in range(self.L):
#                     torch.nn.init.normal_(self.lin_hidden[i].bias)
                    torch.nn.init.kaiming_normal_(self.conv_hidden[i].weight, nonlinearity=self.activation_func)
            except:
                print('Unsupported activation function. Using "linear" as non-linearity argument')
#                 torch.nn.init.normal_(self.lin_in.bias)
#                 torch.nn.init.normal_(self.lin_out.bias)
                torch.nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity='linear')
#                 torch.nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity='linear')
                for i in range(self.L):
#                     torch.nn.init.normal_(self.lin_hidden[i].bias) 
                    torch.nn.init.kaiming_normal_(self.conv_hidden[i].weight, nonlinearity='linear')       
        elif init_type == 'xavier_normal':
#             torch.nn.init.normal_(self.lin_in.bias)
#             torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.xavier_normal_(self.conv_in.weight)
#             torch.nn.init.xavier_normal_(self.conv_out.weight)
            for i in range(self.L):
#                 torch.nn.init.normal_(self.lin_hidden[i].bias)
                torch.nn.init.xavier_normal_(self.conv_hidden[i].weight)
        elif init_type == 'all_const':
#             torch.nn.init.constant_(self.lin_in.bias, kwargs[0])
#             torch.nn.init.constant_(self.lin_out.bias, kwargs[0])
            torch.nn.init.constant_(self.conv_in.weight, kwargs[0])
#             torch.nn.init.constant_(self.conv_out.weight, kwargs[0])
            for i in range(self.L):
#                 torch.nn.init.constant_(self.lin_hidden[i].bias, kwargs[0])
                torch.nn.init.constant_(self.conv_hidden[i].weight, kwargs[0])
        elif init_type == 'orthogonal':
#             torch.nn.init.normal_(self.lin_in.bias)
#             torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.orthogonal_(self.conv_in.weight)
#             torch.nn.init.orthogonal_(self.conv_out.weight)

            for i in range(self.L):
#                 torch.nn.init.normal_(self.lin_hidden[i].bias)
                torch.nn.init.orthogonal_(self.conv_hidden[i].weight)
        else:
            print('Unknown initialization. Using Kaiming normal initialization with linear as nonlinearity argument')
#             torch.nn.init.normal_(self.lin_in.bias)
#             torch.nn.init.normal_(self.lin_out.bias)
            torch.nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity='linear')
#             torch.nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity='linear')
            for i in range(self.L):
#                 torch.nn.init.normal_(self.lin_hidden[i].bias) 
                torch.nn.init.kaiming_normal_(self.conv_hidden[i].weight, nonlinearity='linear')       
  

                
# # Create Sequential model with L hidden layers and varying number of hidden units
# class Sequential_NN_varying_hidden_units(nn.Module):
#     def __init__(self,d,m,k,L,activation='linear', **kwargs):
#         """
#             d: input dimension
#             m: hidden layer dimension 
#             k: output dimension
#             L: number of hidden layers
#         """
#         super().__init__()
        
#         self.activation_func = activation
        
#         if activation=='linear':
#             self.activ = nn.Identity()
#         elif activation=='relu':
#             self.activ = nn.ReLU()
#         elif activation=='leaky_relu':
#             self.activ = nn.LeakyReLU(negative_slope = kwargs['neg_slope'])
#         elif activation=='gelu':
#             self.activ = nn.GELU()
        
#         self.L = L
#         self.lin_out = nn.Linear(m[-1], k, bias=False)
#         self.lin_in = nn.Linear(d, m[0], bias=False)
        
#         self.lin_hidden = nn.ModuleList()
#         for i in range(self.L):
#             self.lin_hidden.append(nn.Linear(m[i], m[i+1], bias=False))

#         self.sequential = nn.Sequential(self.lin_in)
        
#         for i in range(self.L):
#             self.sequential.append(self.activ)
#             self.sequential.append(self.lin_hidden[i])
        
#         self.sequential.append(self.activ)
#         self.sequential.append(self.lin_out)                
        
        
#     def forward(self, xb):
#         xb = self.sequential(xb)       
#         return xb
    
#     def init_weights(self, init_type, *kwargs):
#         if init_type == 'kaiming_normal':
#             try:
#                 torch.nn.init.normal_(self.lin_in.bias)
#                 torch.nn.init.normal_(self.lin_out.bias)
#                 torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity=self.activation_func)
#                 torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity=self.activation_func)
#                 for i in range(self.L):
#                     torch.nn.init.normal_(self.lin_hidden[i].bias)
#                     torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity=self.activation_func)
#             except:
#                 print('Unsupported activation function. Using "linear" as non-linearity argument')
#                 torch.nn.init.normal_(self.lin_in.bias)
#                 torch.nn.init.normal_(self.lin_out.bias)
#                 torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
#                 torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
#                 for i in range(self.L):
#                     torch.nn.init.normal_(self.lin_hidden[i].bias) 
#                     torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')       
#         elif init_type == 'xavier_normal':
#             torch.nn.init.normal_(self.lin_in.bias)
#             torch.nn.init.normal_(self.lin_out.bias)
#             torch.nn.init.xavier_normal_(self.lin_in.weight)
#             torch.nn.init.xavier_normal_(self.lin_out.weight)
#             for i in range(self.L):
#                 torch.nn.init.normal_(self.lin_hidden[i].bias)
#                 torch.nn.init.xavier_normal_(self.lin_hidden[i].weight)
#         elif init_type == 'all_const':
#             torch.nn.init.constant_(self.lin_in.bias, kwargs[0])
#             torch.nn.init.constant_(self.lin_out.bias, kwargs[0])
#             torch.nn.init.constant_(self.lin_in.weight, kwargs[0])
#             torch.nn.init.constant_(self.lin_out.weight, kwargs[0])
#             for i in range(self.L):
#                 torch.nn.init.constant_(self.lin_hidden[i].bias, kwargs[0])
#                 torch.nn.init.constant_(self.lin_hidden[i].weight, kwargs[0])
#         else:
#             print('Unknown initialization. Using Kaiming normal initialization with linear as nonlinearity argument')
#             torch.nn.init.normal_(self.lin_in.bias)
#             torch.nn.init.normal_(self.lin_out.bias)
#             torch.nn.init.kaiming_normal_(self.lin_in.weight, nonlinearity='linear')
#             torch.nn.init.kaiming_normal_(self.lin_out.weight, nonlinearity='linear')
#             for i in range(self.L):
#                 torch.nn.init.normal_(self.lin_hidden[i].bias) 
#                 torch.nn.init.kaiming_normal_(self.lin_hidden[i].weight, nonlinearity='linear')                        
