import numpy as np
import os
import torch

'''
Save results
'''
def make_output_dir(config):
    '''Create a directory to save results'''
    output_dir = f'results/{config.distribution}'
    os.makedirs(output_dir, exist_ok = True)
    return output_dir

def save_score_model(config, score):
    '''Save the score model'''
    save_path = make_output_dir(config) + '/score_{}.pth'.format(config.id)
    torch.save(score.state_dict(), save_path)
    print('Score model saved.')

def save_2d_visualization_data(config, sampler, size_visualize = 5000):
    '''Generate sample and save the first two dimensions for plot'''
    with torch.no_grad():
        gen_data = sampler(size_visualize).detach().cpu().numpy()
    gen_data_2d = gen_data[:, :2]
    
    save_path = make_output_dir(config) + '/gen_data_2d_{}.npz'.format(config.id)
    np.savez(save_path, gen_data_2d = gen_data_2d)

def save_evaluation(config, evaluation):
    '''Save the evaluation reults'''
    save_path = make_output_dir(config) + '/evaluation_{}.csv'.format(config.id)
    np.savetxt(save_path, evaluation, delimiter = ",")
    print('Evaluation saved. The sumary is as follows:')
    print('Mean:', np.mean(evaluation, axis = 0))
    print('Std:', np.std(evaluation, axis = 0))

'''
2D visualization utils
'''
def create_grid_data(x_interval = [-5, 5], y_interval = [-5, 5], precision = 100):
    '''Create grid data for plot'''
    x_n = int((x_interval[1] - x_interval[0])*precision) + 1 # divide unit(say, interval [0,1]) into 'precision' (say, 100) subintervals
    y_n = int((y_interval[1] - y_interval[0])*precision) + 1
    line_x = np.linspace(x_interval[0], x_interval[1], x_n)
    line_y = np.linspace(y_interval[0], y_interval[1], y_n)
    mesh_x, mesh_y = np.meshgrid(line_x, line_y)
    ravel_x = mesh_x.ravel()
    ravel_y = mesh_y.ravel()
    data_grid = np.vstack((ravel_x, ravel_y)).T # shape: (x_n*y_n, 2)
    return data_grid

def prepare_2d_visualization(config, real_model, size_visualize = 5000):
    '''Create real 2D pdf and data for plot, only do once'''
    print('Checking real 2D pdf and data for visualization.')
    file_path = make_output_dir(config) + '/2d_visualization.npz'
    if os.path.isfile(file_path):
        print('Real 2D pdf and data exist.')
    else:
        grid_data = create_grid_data()
        real_pdf = real_model.blocks[0].pdf(grid_data)

        real_data = real_model.rvs(size_visualize)
        real_data_2d = real_data[:, :2]

        np.savez(file_path, real_pdf = real_pdf, real_data_2d = real_data_2d)
        print('Real pdf and data do not exist, have created and saved.')

'''
Miscellaneous utilities
'''
class AttrDict(dict):
    '''Allow accessing dictionary items using attribute style recursively.'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttrDict(v)
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    
    def __setattr__(self, name, value):
        self[name] = value

def set_seed(seed):
    '''Set random seed'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True