from argparse import ArgumentParser
import numpy as np
import time
import torch
import torch.optim as optim
import yaml

import setup, models, networks, evaluation, utils

'''
Score network training
'''
def implicit_score_matching(detail):
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
            if evaluate_score:
                print(f'Score matching [iteration]: {iter + 1}, [loss]: {loss}, [error]: {score_evaluator_val(score_model)}, [time]: {time.time() - start_time}.')
            else:
                print(f'Score matching [iteration]: {iter + 1}, [loss]: {loss}, [time]: {time.time() - start_time}.')

    print('Score model training finshed. Time: {}.'.format(time.time() - start_time))
    if detail:
        utils.save_score_model(config, score_model)

    return score_model

'''
Load experiment configuration
'''
def load_config():
    parser = ArgumentParser(description = 'Evaluate ISM-Diffusion using autograd or Hutchinson trace estimator.')
    parser.add_argument('--distribution', type = str, default = 'polygon_gmm', help = 'Type of target distribution, can be \'polygon_gmm\', \'gauss_push\', or \'energy_model\'.')
    parser.add_argument('--id', type = str, default = 0, help = 'Configuration identifier.')
    args = parser.parse_args()

    with open(f'configs/{args.distribution}/{args.id}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config.update(vars(args))

    method = config['method']
    print(f'Evaluating ISM-Diffusion using {method} on {args.distribution} distribution.')
    return utils.AttrDict(config)

'''
Main function for experiment
'''
if  __name__ == '__main__':
    start_time = time.time()

    # Load configuration
    config = load_config()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    # Set random seed
    utils.set_seed(config.experiment.seed)

    '''
    Model specification
    '''
    # Forward process setting
    diffuse = setup.OUForwardProcess(config.diffusion.beta)

    # Target distribution model
    visualize = False
    evaluate_score = False

    if config.distribution == 'polygon_gmm':
        real_model = models.BlockPolygonGMM(n_blocks = int(config.dim/2))

        visualize = True
        utils.prepare_2d_visualization(config, real_model)

        evaluate_score = True
        oracle_score = models.BlockPolygonGMM_OUScore(real_model, diffuse)

    elif config.distribution == 'gauss_push':
        # Need to specify type of invertible mapping, we use TanhLinearMap
        invertible_map = models.TanhLinearMap(models.get_A_matrix(config))
        real_model = models.GaussianPushforward(invertible_map)

    elif config.distribution == 'energy_model':
        # Need to specify type of energy, we use QuarticEnergy
        real_model = models.QuarticEnergy(models.get_Q_matrix(config), config.model.alpha)

    else:
        raise ValueError('Wrong distribution model name: the distribution should be \'polygon_gmm\', \'gauss_push\', or \'energy_model\'.')
    
    print('Model specified and training starts. Time: {}.'.format(time.time() - start_time))

    '''
    Score model training and evaluation
    '''
    evaluation_results = []
    if evaluate_score:
        score_evaluator_val = evaluation.get_score_est_error_evaluator('validation', real_model, oracle_score, diffuse, config, device)
        score_evaluator_test = evaluation.get_score_est_error_evaluator('test', real_model, oracle_score, diffuse, config, device)
    dist_evaluator_w1 = evaluation.get_dist_est_error_evaluator('w1', real_model, config.eva)
    dist_evaluator_mmd = evaluation.get_dist_est_error_evaluator('mmd', real_model, config.eva)

    for repetition in range(config.experiment.num_repetitions):
        # ---------
        # Training
        # ---------
        score = implicit_score_matching(repetition == 0)
        sampler = setup.get_sampler(score, config, device = device)

        # ------------
        # Evaluation
        # ------------
        w1_distance = dist_evaluator_w1(sampler)
        mmd_distance = dist_evaluator_mmd(sampler)

        if visualize:
            utils.save_2d_visualization_data(config, sampler)
        
        if evaluate_score:
            score_error = score_evaluator_test(score)
            evaluation_results.append([w1_distance, mmd_distance, score_error])
        else:
            evaluation_results.append([w1_distance, mmd_distance])

    # Save results
    utils.save_evaluation(config, evaluation_results)
    print('Experiment finished. Total time:', time.time() - start_time)