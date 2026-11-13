import torch
import torch.nn as nn

'''
RePU FNN
'''
class RePU_activation(nn.Module):
    '''
    RePU activation function: RePU_p(x) = max(0, x)^p
    '''
    def __init__(self, p = 2):
        super(RePU_activation, self).__init__()
        self.p = p

    def forward(self, input):
        return torch.relu(input) ** self.p

class RePU_FNN(nn.Module):
    '''
    Model network: a RePU-activated feedforward neural network for score estimation.
    Depth = nd, width = nw.
    Params:
        dim: input dimension
        nd: depth (number of hidden layers)
        nw: width
        p: order of RePU, default 2
    '''
    def __init__(self, dim: int, nd: int, nw: int, p = 2):
        super(RePU_FNN, self).__init__()
        # Network
        modules = []
        in_dim = dim + 1 # First layer input dim

        for _ in range(nd):
            modules.append(nn.Sequential(nn.Linear(in_dim, nw), RePU_activation(p)))
            in_dim = nw

        # Final linear layer: output score in R^dim
        self.main = nn.Sequential(*modules, nn.Linear(nw, dim))

    def forward(self, input):
        return self.main(input)

'''
Differentiable RePU FNN
'''
class RePU_layer(nn.Module):
    '''
    Linear layer followed by RePU_p activation.
    Stores pre-activation for analytic differentiation.
    '''
    def __init__(self, in_dim, out_dim, p = 2, init_scale = 0.1):
        super(RePU_layer, self).__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * init_scale)
        self.b = nn.Parameter(torch.zeros(out_dim))
        self.p = p

    def forward(self, x):
        z = x @ self.W.T + self.b
        return torch.relu(z) ** self.p, z

    def derivative(self, z):
        # d/dz RePU_p(z)
        return self.p * (torch.relu(z) ** (self.p - 1))

class RePU_FNN_exactgrad(nn.Module):
    '''
    RePU-activated FNN with analytic gradient and divergence.
    Input is (x, t), but derivatives are taken w.r.t x only.
    '''
    def __init__(self, dim, nd, nw, p = 2):
        super().__init__()
        self.dim = dim
        self.p = p

        # Network
        modules = []
        in_dim = dim + 1 # x and t

        for _ in range(nd):
            modules.append(RePU_layer(in_dim, nw, p))
            in_dim = nw

        self.hidden = nn.ModuleList(modules)
        self.out = nn.Linear(nw, dim)

    # Forward
    def forward(self, input):
        '''
        input: (n, dim + 1)
        '''
        h = input
        self.z_list = []     # pre-activations

        for layer in self.hidden:
            h, z = layer(h)
            self.z_list.append(z)

        return self.out(h)
    
    def gradient(self, input):
        '''
        Compute Jacobian ∇_x f(x,t) analytically.
        Return: (n, dim, dim)
        '''
        n = input.shape[0]
        dim = self.dim
        device = input.device

        # Forward pass must be called first
        _ = self.forward(input)

        # Start with ∂h_0/∂x, the gradient of concatenate opertaion
        J = torch.zeros(n, dim + 1, dim, device = device)
        J[:, :self.dim, :] = torch.eye(dim, device = device).unsqueeze(0)

        # Forward prop through RePU layers
        for layer, z in zip(self.hidden, self.z_list):
            D = layer.derivative(z)                  # (n, out_dim)
            W = layer.W                              # (out_dim, in_dim)
            
            # J <- diag(D) W J
            J = torch.einsum(
                'ni,ij,njk->nik',
                D, W, J
            )

        # output layer
        # J <- W_out J
        J = torch.einsum(
            'ij,njk->nik',
            self.out.weight, J
        )

        return J

    def divergence(self, input):
        '''
        Compute div_x f(x,t) exactly.
        Returns: (n,)
        '''
        J = self.gradient(input)
        return torch.einsum('nii->n', J)

'''
ReLU FNN
'''
class ReLU_FNN(nn.Module):
    '''
    Model network: a ReLU-activated feedforward neural newtork for score estimation.
    Depth = nd, width = nw.
    Params:
        dim: input dimension
        nd: depth (number of hidden layers)
        nw: width
        leak: parameter of LeakyReLU activation, default 0.1
    '''
    def __init__(self, dim: int, nd: int, nw: int, leak = 0.1):
        super(ReLU_FNN, self).__init__()
        # Network
        modules = []
        in_dim = dim + 1

        for _ in range(nd):
            modules.append(nn.Sequential(nn.Linear(in_dim, nw), nn.LeakyReLU(leak, inplace = True)))
            in_dim = nw # First layer input dim

        self.main = nn.Sequential(*modules, nn.Linear(nw, dim))

    def forward(self, input):
        return self.main(input)