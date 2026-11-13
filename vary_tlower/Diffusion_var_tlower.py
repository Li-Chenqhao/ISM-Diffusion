from argparse import ArgumentParser
import copy
import numpy as np
import time
import torch
import torch.optim as optim
import yaml

import setup, models, networks, evaluation, utils

'''
Specification
'''
# Grid of t_lotwer
TLOWER_LIST = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

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
def implicit_score_matching_exactgrad(config, detail):
    score_model = networks.RePU_FNN_exactgrad(config.dim, config.net.nd, config.net.nw, config.net.p).to(device)
    loss_func = setup.ISM_loss_exactgrad(score_model)
    optimizer = optim.Adam(score_model.parameters(), lr = config.train.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.train.step, gamma = config.train.gamma)

    for iter in range(config.train.num_iters):
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

        if detail and (iter + 1) % config.train.log_step == 0:
            print(f'Score matching [iteration]: {iter + 1}, [loss]: {loss}, [error]: {score_evaluator_val(score_model)}, [time]: {time.time() - start_time}.')

    print('Score model training finshed. Time: {}.'.format(time.time() - start_time))
    if detail:
        utils.save_score_model(config, score_model)

    return score_model

def implicit_score_matching(config, detail):
    loss_func = setup.ISM_loss(config, device)
    score_model = networks.RePU_FNN(config.dim, config.net.nd, config.net.nw, config.net.p).to(device)
    optimizer = optim.Adam(score_model.parameters(), lr = config.train.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.train.step, gamma = config.train.gamma)

    for iter in range(config.train.num_iters):
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

        if detail and (iter + 1) % config.train.log_step == 0:
            print(f'Score matching [iteration]: {iter + 1}, [loss]: {loss}, [error]: {score_evaluator_val(score_model)}, [time]: {time.time() - start_time}.')

    print('Score model training finshed. Time: {}.'.format(time.time() - start_time))
    if detail:
        utils.save_score_model(config, score_model)

    return score_model

def denoising_score_matching(config, detail):
    loss_func = torch.nn.MSELoss()

    if config.net.type == 'relu':
        score_model = networks.ReLU_FNN(config.dim, config.net.nd, config.net.nw, config.net.leak).to(device)
    elif config.net.type == 'repu':
        score_model = networks.RePU_FNN(config.dim, config.net.nd, config.net.nw, config.net.p).to(device)
    else:
        raise ValueError('Wrong network type: the net_type should be \'relu\' or \'repu\'.')

    optimizer = optim.Adam(score_model.parameters(), lr = config.train.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.train.step, gamma = config.train.gamma)

    for iter in range(config.train.num_iters):
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

        if detail and (iter + 1) % config.train.log_step == 0:
            print(f'Score matching [iteration]: {iter + 1}, [loss]: {loss}, [error]: {score_evaluator_val(score_model)}, [time]: {time.time() - start_time}.')

    print('Score model training finshed. Time: {}.'.format(time.time() - start_time))
    if detail:
        utils.save_score_model(config, score_model)

    return score_model

'''
Load experiment configuration
'''
def load_base_config():
    parser = ArgumentParser(description = 'Evaluating different methods with varying t_lower on Polygon GMM distribution.')
    parser.add_argument('--distribution', type = str, default = 'polygon_gmm_vartlower', help = 'Save path for the results.')
    parser.add_argument('--id', type = str, default = 'main', help = 'Configuration identifier.')
    args = parser.parse_args()

    with open(f'configs/var_tlower_{args.id}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config.update(vars(args))

    print('Evaluating different methods with varying t_lower on Polygon GMM distribution.')
    return utils.AttrDict(config)

'''
Main function for experiment
'''
if  __name__ == '__main__':
    start_time = time.time()

    # Load configuration
    base_config = load_base_config()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    # Set random seed
    utils.set_seed(base_config.experiment.seed)

    '''
    Model specification
    '''
    # Forward process setting
    diffuse = setup.OUForwardProcess(base_config.diffusion.beta)

    # Target distribution model
    real_model = models.BlockPolygonGMM(n_blocks = int(base_config.dim/2))
    # utils.prepare_2d_visualization(base_config, real_model)
    oracle_score = models.BlockPolygonGMM_OUScore(real_model, diffuse)
    print('Model specified and training starts. Time: {}.'.format(time.time() - start_time))

    '''
    Score model training and evaluation
    '''
    # Define evaluators
    score_evaluator_val = evaluation.get_score_est_error_evaluator('validation', real_model, oracle_score, diffuse, base_config, device)
    score_evaluator_test = evaluation.get_score_est_error_evaluator('test', real_model, oracle_score, diffuse, base_config, device)
    dist_evaluator_w1 = evaluation.get_dist_est_error_evaluator('w1', real_model, base_config.eva)
    dist_evaluator_mmd = evaluation.get_dist_est_error_evaluator('mmd', real_model, base_config.eva)

    for method, info in list(METHODS.items()):
        tag = info['tag']
        config = copy.deepcopy(base_config)
        config.method = info['method']
        config.net.type = info['network']

        print(f'Evaluating method {method}. Time: {time.time() - start_time}.')

        for t_lower in TLOWER_LIST:
            config.diffusion.t_lower = t_lower
            config.id = f'{base_config.id}_{tag}_{t_lower}'
            evaluation_results = []

            for repetition in range(config.experiment.num_repetitions):
                # ---------
                # Training
                # ---------
                if method == 'ISM_ExactGrad':
                    score = implicit_score_matching_exactgrad(config, repetition == 0)
                elif method == 'ISM_AutoGrad' or method == 'ISM_Hutchinson':
                    score = implicit_score_matching(config, repetition == 0)
                else:
                    score = denoising_score_matching(config, repetition == 0)

                sampler = setup.get_sampler(score, config, device = device)

                # ------------
                # Evaluation
                # ------------
                w1_distance = dist_evaluator_w1(sampler)
                mmd_distance = dist_evaluator_mmd(sampler)

                utils.save_2d_visualization_data(config, sampler)
                
                score_error = score_evaluator_test(score)
                evaluation_results.append([w1_distance, mmd_distance, score_error])

            # Save results
            utils.save_evaluation(config, evaluation_results)
            print('Experiment finished. Total time:', time.time() - start_time)