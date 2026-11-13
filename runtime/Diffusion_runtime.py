import copy
import numpy as np
import time
import torch
import torch.optim as optim
import yaml

import setup, models, networks, utils

'''
Specification
'''
# Grid of dimension
DIM_LIST = [8, 12, 16, 20]

# List of methods
METHODS = {
    "ISM_AutoGrad": {
        "tag": "A",
        "method": "autograd",
        "network": "repu"
    },
    "ISM_Hutchinson": {
        "tag": "H",
        "method": "hutchinson",
        "network": "repu"
    },
    "ISM_ExactGrad": {
        "tag": "E",
        "method": None,
        "network": "repu"
    },
    "DSM_ReLU": {
        "tag": "DL",
        "method": None,
        "network": "relu"
    },
    "DSM_RePU": {
        "tag": "DP",
        "method": None,
        "network": "repu"
    }
}

'''
Score network training
'''
def implicit_score_matching_exactgrad(config):
    score_model = networks.RePU_FNN_exactgrad(config.dim, config.net.nd, config.net.nw, config.net.p).to(device)
    loss_func = setup.ISM_loss_exactgrad(score_model)
    optimizer = optim.Adam(score_model.parameters(), lr = config.train.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.train.step, gamma = config.train.gamma)

    torch.cuda.synchronize()
    iter_start_time = time.time()

    for _ in range(config.train.num_iters):
        score_model.zero_grad(set_to_none = True)

        batch = real_model.rvs(config.train.bs)
        t = np.random.uniform(config.diffusion.t_lower * config.diffusion.t_factor, config.diffusion.t_upper, (config.train.bs, 1))
        noisy_data = diffuse.sample_xt(batch, t)
        model_input = np.concatenate((noisy_data, t), axis = 1)
        model_input = torch.from_numpy(model_input).float().to(device)

        score = score_model(model_input)
        loss = loss_func(score, model_input)
        loss.backward()
        optimizer.step()
        scheduler.step()

    torch.cuda.synchronize()
    return time.time() - iter_start_time

def implicit_score_matching(config):
    loss_func = setup.ISM_loss(config, device)
    score_model = networks.RePU_FNN(config.dim, config.net.nd, config.net.nw, config.net.p).to(device)
    optimizer = optim.Adam(score_model.parameters(), lr = config.train.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.train.step, gamma = config.train.gamma)

    torch.cuda.synchronize()
    iter_start_time = time.time()

    for _ in range(config.train.num_iters):
        score_model.zero_grad(set_to_none = True)

        batch = real_model.rvs(config.train.bs)
        t = np.random.uniform(config.diffusion.t_lower * config.diffusion.t_factor, config.diffusion.t_upper, (config.train.bs, 1))
        noisy_data = diffuse.sample_xt(batch, t)
        noisy_data = torch.from_numpy(noisy_data).float().to(device)
        noisy_data.requires_grad_(True)
        t = torch.from_numpy(t).float().to(device)
        model_input = torch.cat((noisy_data, t), dim = 1)

        score = score_model(model_input)
        loss = loss_func(score, noisy_data)
        loss.backward()
        optimizer.step()
        scheduler.step()

    torch.cuda.synchronize()
    return time.time()- iter_start_time

def denoising_score_matching(config):
    loss_func = torch.nn.MSELoss()

    if config.net.type == 'relu':
        score_model = networks.ReLU_FNN(config.dim, config.net.nd, config.net.nw, config.net.leak).to(device)
    elif config.net.type == 'repu':
        score_model = networks.RePU_FNN(config.dim, config.net.nd, config.net.nw, config.net.p).to(device)
    else:
        raise ValueError('Wrong network type: the net_type should be \'relu\' or \'repu\'.')

    optimizer = optim.Adam(score_model.parameters(), lr = config.train.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.train.step, gamma = config.train.gamma)

    torch.cuda.synchronize()
    iter_start_time = time.time()

    for _ in range(config.train.num_iters):
        score_model.zero_grad(set_to_none = True)

        batch = real_model.rvs(config.train.bs)
        t = np.random.uniform(config.diffusion.t_lower * config.diffusion.t_factor, config.diffusion.t_upper, (config.train.bs, 1))
        noisy_data = diffuse.sample_xt(batch, t)
        model_input = np.concatenate((noisy_data, t), axis = 1)
        model_input = torch.from_numpy(model_input).float().to(device)

        train_target = diffuse.conditional_score(noisy_data, batch, t)
        train_target = torch.from_numpy(train_target).float().to(device)
        loss = loss_func(score_model(model_input), train_target)
        loss.backward()
        optimizer.step()
        scheduler.step()

    torch.cuda.synchronize()
    return time.time() - iter_start_time

'''
Load experiment configuration
'''
def load_config(dim):
    with open(f'configs/runtime_{dim}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['id'] = ''
    config['distribution'] = 'polygon_gmm_runtime'
    
    return utils.AttrDict(config)

'''
Main function for experiment
'''
if  __name__ == '__main__':
    print('Evaluating running times of different methods with varying dimension on Polygon GMM distribution.')
    start_time = time.time()

    # Device and software
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)
    print('PyTorch version:', torch.__version__)
    print('CUDA version (PyTorch build):', torch.version.cuda)
    if device.type == 'cuda':
        print('GPU model:', torch.cuda.get_device_name(device))

    for dim in DIM_LIST:
        print('Evaluating for dimension: {}.'.format(dim))
        # Load configuration
        dim_config = load_config(dim)

        # Set random seed
        utils.set_seed(dim_config.experiment.seed)

        '''
        Model specification
        '''
        # Forward process setting
        diffuse = setup.OUForwardProcess(dim_config.diffusion.beta)

        # Target distribution model
        real_model = models.BlockPolygonGMM(n_blocks = int(dim_config.dim/2))
        print('Model specified and training starts. Time: {}.'.format(time.time() - start_time))

        '''
        Score model training
        '''
        for method, info in list(METHODS.items()):
            tag = info['tag']
            config = copy.deepcopy(dim_config)
            config.id = f'{tag}_{dim}'
            config.method = info['method']
            config.net.type = info['network']

            print(f'Evaluating method {method}. Time: {time.time() - start_time}.')

            # ---------
            # Training
            # ---------
            if method == 'ISM_ExactGrad':
                method_time = implicit_score_matching_exactgrad(config)
            elif method == 'ISM_AutoGrad' or method == 'ISM_Hutchinson':
                method_time = implicit_score_matching(config)
            else:
                method_time = denoising_score_matching(config)

            runtime_per_iter = method_time / config.train.num_iters
            print(f'Score model training finshed. [Total time]: {method_time}, [Avergage time]: {runtime_per_iter}.')

            # Save result
            utils.save_evaluation(config, [runtime_per_iter])
    print('Experiment finished. Total time:', time.time() - start_time)