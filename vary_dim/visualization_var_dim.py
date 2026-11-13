import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Specifications
# ----------------------------
# Result directory
RESULTS_DIR = 'results/polygon_gmm_vardim'

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
    'w1': 'Wasserstein-$1$ distance',
    'mmd': 'Squared MMD',
    'score':'$L^{2}$ score estimation error',
}

# 2D plot settings
PLOT_XLIM = [-5, 5]
PLOT_YLIM = [-5, 5]

# Plot settings
TITLE_FONT = 14
LABEL_FONT = 14
FIGSIZE_FACTOR = 3.0

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

        # Take the mean over repetitions
        mean_values = np.mean(data, axis = 0)

        # Save data
        for i, measure in enumerate(MEASURES.keys()):
            if method not in eva_data[measure]:
                eva_data[measure][method] = []
            eva_data[measure][method].append((dim, mean_values[i]))

# 2D data
# Generated data
for dim in DIM_LIST:
    for method, tag in METHODS.items():
        data_path = f'{RESULTS_DIR}/gen_data_2d_{tag}_{str(dim)}.npz'
        data_list.append(np.load(data_path)['gen_data_2d'])

# ----------------------------
# Line chart
# ----------------------------
for measure, measure_fullname in MEASURES.items():
    plt.figure(figsize=(7, 6))

    for method in METHODS.keys():
        method_data = eva_data[measure][method]
        d_vals = [x[0] for x in method_data]
        y_vals = [x[1] for x in method_data]

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

# ----------------------------
# 2D data
# ----------------------------
# Subplot titles
titles = list(METHODS.keys())

num_rows = len(DIM_LIST)
num_columns = len(titles)
figsize = (num_columns * FIGSIZE_FACTOR, num_rows * FIGSIZE_FACTOR)

fig, axes = plt.subplots(
    nrows = num_rows,
    ncols = num_columns,
    figsize = figsize,
    constrained_layout = True
)

# Rows: generated data
idx = 0  # index in data_list
for row in range(num_rows):
    # Set row label on leftmost subplot
    axes[row, 0].set_ylabel(f'Dim: {DIM_LIST[row]}', fontsize = TITLE_FONT)

    for col in range(num_columns):
        ax = axes[row, col]
        data = data_list[idx]
        idx += 1

        ax.scatter(
            data[:, 0],
            data[:, 1],
            s = 5,
            alpha = 0.6
        )
        ax.set_xlim(PLOT_XLIM)
        ax.set_ylim(PLOT_YLIM)
        ax.set_aspect('equal')

        # Title only for first row of generated samples
        if row == 0:
            ax.set_title(titles[col], fontsize = TITLE_FONT)

save_path = f'results/2dim_visualization.{FIG_FORMAT}'
plt.savefig(save_path)
plt.close()

print(f'Saved figure: {save_path}')