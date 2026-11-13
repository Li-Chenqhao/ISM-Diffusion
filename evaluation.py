import numpy as np
import os
import ot
import torch

from setup import OUForwardProcess

'''
Evaluate score estimation error
'''
def get_score_est_error_evaluator(mode: str, real_model, oracle_score, diffuse: OUForwardProcess, config, device):
    '''
    Factory function that returns a callable evaluator for score estimation error.
    Params:
        mode : either "validation" or "test", determines sample size.
        real_model: target distribution object.
        diffuse: OU forward process object.
        config: configuration object.
        device: a torch.device object.
    '''
    # Sample size
    if mode == 'validation':
        size = config.eva.score_val_size
    elif mode == 'test':
        size = config.eva.score_test_size
    else:
        raise ValueError('Wrong mode name: the mode should be \'validation\', or \'test\'.')

    # Sample X0 from reak model and t from Uniform[T_lower, T_upper]
    x0 = real_model.rvs(size)
    t = np.random.uniform(config.diffusion.t_lower, config.diffusion.t_upper, size = (size, 1))

    # Forward diffusion
    xt = diffuse.sample_xt(x0, t)

    # True score
    true_score = oracle_score.score(xt, t)
    true_score = torch.from_numpy(true_score).float().to(device)

    # Concatenate x_t and t to get model input
    model_input = np.concatenate((xt, t), axis = 1)
    model_input = torch.from_numpy(model_input).float().to(device)
    
    # ------------------
    # Returned evaluator
    # ------------------
    def evaluator(score_model):
        '''
        Compute score estimation error:
        E || s_theta(X_t, t) - s_0(X_t, t) ||_2^2
        '''
        with torch.no_grad():
            est_score = score_model(model_input)
            error = torch.mean(torch.sum((est_score - true_score) ** 2, dim = 1))

        return error.item()

    return evaluator

'''
Evaluate distribution estimation error
'''
def wasserstein_1_distance(x, y, reg = 0.05):
    '''
    Approximate Wasserstein-1 distance using Sinkhorn algorithm.
    Params:
        x: Samples from true distribution, shape (n, d).
        y: Samples from learned distribution, shape (m, d).
        reg: Entropic regularization parameter.
    '''
    n, m = x.shape[0], y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m

    cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis = 2)
    return ot.sinkhorn2(a, b, cost, reg)

def wasserstein_1_distance_stable(x, y, reg = 2.0, cost_clip = 1e2):
    '''
    Approximate Wasserstein-1 distance using Sinkhorn algorithm, specifically improved computational stability.
    Params:
        x: Samples from true distribution, shape (n, d).
        y: Samples from learned distribution, shape (m, d).
        reg: Entropic regularization parameter, choosing 1.0 makes the computation more stable
        cost_clip: Clip large values of cost to this value.
    '''
    n, m = x.shape[0], y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m

    cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis = 2)
    cost = np.clip(cost, 0, cost_clip)   # Prevent extremely large cost

    return ot.sinkhorn2(a, b, cost, reg)

def mmd_distance(x, y, bandwidth = None):
    '''
    Squared MMD distance with Gaussian kernel.
    Params:
        x: Samples from true distribution, shape (n, d).
        y: Samples from learned distribution, shape (m, d).
        bandwidth: Kernel bandwidth. If None, use median heuristic.
    '''
    def _gaussian_kernel(x, y, bandwidth):
        diff = x[:, None, :] - y[None, :, :]
        return np.exp(-np.sum(diff ** 2, axis = 2) / (2.0 * bandwidth ** 2))

    if bandwidth is None:
        z = np.vstack([x, y])
        dists = np.linalg.norm(z[:, None, :] - z[None, :, :], axis = 2)
        bandwidth = np.median(dists)

    k_xx = _gaussian_kernel(x, x, bandwidth)
    k_yy = _gaussian_kernel(y, y, bandwidth)
    k_xy = _gaussian_kernel(x, y, bandwidth)

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

def mmd_distance_stable(x, y, bandwidth = None, diff_clip = 1e2, bandwidth_clip = 1e-1):
    '''
    Squared MMD distance with Gaussian kernel, specifically improved computational stability.
    Params:
        x: Samples from true distribution, shape (n, d).
        y: Samples from learned distribution, shape (m, d).
        bandwidth: Kernel bandwidth. If None, use median heuristic.
        diff_clip: Prevent the difference and its square in the calculation of gaussian kernel to be too large.
        bandwidth_clip: Prevent the bandwidth to be too small.
    '''
    def _gaussian_kernel(x, y, h):
        diff = x[:, None, :] - y[None, :, :]
        diff = np.clip(diff, -diff_clip, diff_clip)
        sq = np.sum(diff ** 2, axis = 2)
        # sq = np.clip(sq, 0, diff_clip ** 2) # Prevent exp(-inf)
        return np.exp(-sq / (2 * h * h))

    if bandwidth is None:
        z = np.vstack([x, y])
        d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis = 2)
        bandwidth = np.median(d)
        bandwidth = max(bandwidth, bandwidth_clip) # Restrict the lower limit of bandwidth

    k_xx = _gaussian_kernel(x, x, bandwidth)
    k_yy = _gaussian_kernel(y, y, bandwidth)
    k_xy = _gaussian_kernel(x, y, bandwidth)

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

def get_dist_est_error_evaluator(metric: str, real_model, config_eva, gen_clip = 1e2):
    '''
    Construct a distribution estimation error evaluator.
    Params:
        metric: Distribution discrepancy measure to use. Either 'w1' or 'mmd'.
        real_model : Target distribution object with method real_model.rvs(n).
        config : Configuration object.
    '''
    if metric == 'w1':
        true_size = config_eva.w1_size_true
        gen_size = config_eva.w1_size_gen
    elif metric == 'mmd':
        true_size = config_eva.mmd_size_true
        gen_size = config_eva.mmd_size_gen
    else:
        raise ValueError('Wrong metric name: the metric should be \'w1\', or \'mmd\'.')

    # Sample from true distribution
    x_true = real_model.rvs(true_size)
    x_true = np.asarray(x_true)

    def evaluator(sampler):
        '''
        sampler: Callable generated by get_sampler.
        '''
        # Sample from diffusion model
        with torch.no_grad():
            x_gen = sampler(gen_size)
        x_gen = x_gen.detach().cpu().numpy()

        # Prevent overflow
        x_gen = np.nan_to_num(x_gen, posinf = gen_clip, neginf = -gen_clip)
        x_gen = np.clip(x_gen, -gen_clip, gen_clip)

        # ------------------
        # Compute distance
        # ------------------
        if metric == 'w1':
            dist = wasserstein_1_distance_stable(
                x_true,
                x_gen,
                reg = config_eva.w1_reg,
            )
        elif metric == 'mmd':
            dist = mmd_distance_stable(
                x_true,
                x_gen,
                bandwidth = getattr(config_eva, 'mmd_bandwidth', None),
            )
            
        del x_gen
        return dist

    return evaluator