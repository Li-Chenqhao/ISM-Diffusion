import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Specifications
# ----------------------------
# Result directory
RESULTS_DIR = 'results/polygon_gmm_runtime'

# Figure Format
FIG_FORMAT = 'pdf'

# List of methods
METHODS = {
    'ISM_ExactGrad': 'E',
    'ISM_AutoGrad': 'A',
    'ISM_Hutchinson': 'H',
    'DSM_ReLU': 'DL',
    'DSM_RePU': 'DP',
}

# Grid of dimension
DIM_LIST = [8, 12, 16, 20]

# Measures for evaluating performance
MEASURES = {
    'time': 'Runtime (ms)',
}

# ----------------------------
# Read data and compute mean
# ----------------------------
# Initialization
eva_data = {measure: {} for measure in MEASURES.keys()}
data_list = []

# Evaluation data for line chart 
for method, tag in METHODS.items():
    for dim in DIM_LIST:
        # Read CSV
        data = np.genfromtxt(f'{RESULTS_DIR}/evaluation_{tag}_{str(dim)}.csv', delimiter = ',')

        # Save data
        for measure in MEASURES.keys():
            if method not in eva_data[measure]:
                eva_data[measure][method] = []
            eva_data[measure][method].append((dim, data))

# ----------------------------
# Line chart
# ----------------------------
for measure, measure_fullname in MEASURES.items():
    plt.figure(figsize=(7, 6))

    for method in METHODS.keys():
        method_data = eva_data[measure][method]
        d_vals = [x[0] for x in method_data]
        y_vals = [1000 * x[1] for x in method_data]

        # Line chart
        plt.plot(d_vals, y_vals, marker = 'o', label = method)

    # Set axis
    plt.xlabel('Dimension')
    plt.ylabel(measure_fullname)

    # plt.title(f"{measure} vs dimension")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()

    save_path = f'results/{measure}_vs_dim.{FIG_FORMAT}'
    plt.savefig(save_path)
    plt.close()

    print(f'Saved figure: {save_path}')