import numpy as np
import torch

from networks import RePU_FNN_exactgrad

'''
Diffusion process
'''
class OUForwardProcess:
    '''
    Ornstein–Uhlenbeck forward diffusion with beta_t set to be a constant beta.
    X_t = m_t X_0 + sigma_t ε
    Params:
        beta: constant in OU process
    '''
    def __init__(self, beta = 1.0):
        self.beta = beta

    # Calculate m_t and sigma_t
    def m_t(self, t):
        return np.exp(-self.beta * t)

    def sigma_t(self, t):
        return np.sqrt(1.0 - np.exp(-2 * self.beta * t))

    def sample_xt(self, x0, t):
        '''
        Sample X_t | X_0 = x0
        x0: (n, d)
        t: (n, 1)
        '''
        t = np.asarray(t)
        m = self.m_t(t)
        sigma = self.sigma_t(t)
        z = np.random.randn(*x0.shape)
        return m * x0 + sigma * z

    def conditional_score(self, xt, x0, t):
        '''
        Computes the condistional score ∇ log p_t(X_t | X_0)
        Used as training target in DSM loss
        '''
        m = self.m_t(t)
        sigma2 = self.sigma_t(t) ** 2
        return (m * x0 - xt) / sigma2

def get_sampler(score_model, config, device):
    '''
    Construct a data generator based on the trained score model and the backward SDE.
    The backward SDE is discretized using the Euler–Maruyama scheme.
    '''
    interval = (config.diffusion.t_upper - config.diffusion.t_lower) / config.sampling.sample_step # dt

    def sampler(size):
        # Initial condition: pure Gaussian noise
        sample = torch.randn((size, config.dim), device = device)

        for step in range(config.sampling.sample_step):
            # Reverse-time parameter: t = T - k * dt
            time = config.diffusion.t_upper - step * interval
            time = torch.full((size, 1), time, device = device)

            # Score network input: concatenate (x, t)
            score_input = torch.cat((sample, time), dim = 1)
            score = score_model(score_input)

            # Drift term: beta * (x + 2 * score)
            drift = config.diffusion.beta * (sample + 2.0 * score)

            # Diffusion term: sqrt(2 * beta * dt) * N(0, I)
            noise = torch.randn_like(sample)
            diffusion = (2.0 * config.diffusion.beta * interval) ** 0.5 * noise

            # Euler–Maruyama update
            sample = sample + drift * interval + diffusion

        return sample

    return sampler

'''
Implicit score matching loss functions
'''
def ISM_loss(config, device):
    '''
    Factory function for exact implicit score matching loss computation.
    The key is to compute the divergence: div s(x, t) = sum_i ∂ s_i(x, t) / ∂ x_i
    Params:
        config: Provides method and seed.
            method: Either 'autograd' or 'hutchinson'.
                autograd: Computes the exact divergence via autograd.
                hutchinson: Approximate the divergence using the Hutchinson trace estimator.
    '''
    # Create a dedicated RNG if seed is provided
    if config.method == 'hutchinson' and config.experiment.seed is not None:
        gen = torch.Generator(device = device)
        gen.manual_seed(config.experiment.seed)
    else:
        gen = None

    def loss_func(score, x):
        '''
        Params
        score: Score network output s_theta(x, t), shape (n, d).
        x: Spatial input, shape (n, d), must have requires_grad = True.
        Return: Scalar loss value.
        '''
        # Energy term: 1/2 ||s||^2
        loss_energy = 0.5 * torch.sum(score ** 2, dim = 1)

        # Exact divergence computation using autograd
        if config.method == 'autograd':
            div = 0.0
            for i in range(score.shape[1]):
                grad_i = torch.autograd.grad(
                    outputs = score[:, i].sum(),
                    inputs = x,
                    create_graph = True,
                )[0]
                div = div + grad_i[:, i]

        # Hutchinson trace estimator for divergence
        elif config.method == 'hutchinson':
            # Controlled Hutchinson noise
            if gen is not None:
                eps = torch.randn(score.shape, device = device, dtype = score.dtype, generator = gen)
            else:
                eps = torch.randn_like(score)

            dot = torch.sum(score * eps)

            grad = torch.autograd.grad(
                outputs = dot,
                inputs = x,
                create_graph = True,
            )[0]

            div = torch.sum(grad * eps, dim = 1)

        else:
            raise ValueError('Wrong mode name: the mode should be \'autograd\' or \'hutchinson\'. For \'exactgrad\', please use \'ISM_loss_exactgrad\'.')

        return torch.mean(loss_energy + div)

    return loss_func

def ISM_loss_exactgrad(score_model: RePU_FNN_exactgrad):
    '''
    Factory function for exact implicit score matching loss computation using backpropagation of RePU FNN.
    The key is to comppute the divergence: div s(x, t) = sum_i ∂ s_i(x, t) / ∂ x_i
    Params:
        score_model: a network.RePU_FNN_exactgrad object.
    '''
    def loss_func(score, input):
        '''
        Params
        score: Score network output s_theta(x, t), shape (n, d).
        input: Input of score network (x, t), shape (n, d + 1), no need to have requires_grad = True.
        Return: Scalar loss value.
        '''
        # Energy term: 1/2 ||s||^2
        loss_energy = 0.5 * torch.sum(score ** 2, dim = 1)

        # Exact divergence computation using backpropagation of RePU FNN
        div = score_model.divergence(input)

        return torch.mean(loss_energy + div)

    return loss_func