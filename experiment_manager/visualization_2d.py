import matplotlib.pyplot as plt
import numpy as np
import os

'''
Specification
'''
# Configuration id
CONFIG_ID = 'baseline'

# Methods for score estimation
METHODS = {
    'ISM_ExactGrad': 'E',
    'ISM_AutoGrad': 'A',
    'ISM_Hutchinson': 'H',
    'DSM_ReLU': 'DL',
    'DSM_RePU': 'DP'
}

# Data model for visualization
MODEL = 'polygon_gmm'

# Visualization settings
PLOT_XLIM = [-5, 5]
PLOT_YLIM = [-5, 5]
PLOT_PRECISION = 100

# Plot settings
N_ROWS = 2
N_COLS = 4
FIG_SIZE = (18, 9)
TITLE_SIZE = 14
LABEL_SIZE = 12

'''
Read the data
'''
data_list = []
x_n = int((PLOT_XLIM[1] - PLOT_XLIM[0]) * PLOT_PRECISION) + 1
y_n = int((PLOT_YLIM[1] - PLOT_YLIM[0]) * PLOT_PRECISION) + 1

# Real pdf and data
data = np.load('2d_visualization.npz')
real_pdf = data['real_pdf'].reshape(y_n, x_n)
real_data = data['real_data_2d']

# Generated data from methods
gen_data_list = []
for method, tag in METHODS.items():
    data_path = os.path.join(CONFIG_ID, method, 'results', MODEL, f'gen_data_2d_{CONFIG_ID}_{tag}.npz')
    gen_data_list.append(np.load(data_path)["gen_data_2d"])

'''
plot
'''
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize = FIG_SIZE, constrained_layout = True)

# BASE INDEXING:
# Row1 col0: density
# Row1 col1: real
# Row1 col2–3: method 1–2
# Row2 col1–3: method 3–5

# Remove empty subplot: (1,0)
fig.delaxes(axes[1, 0])

# Density plot
ax = axes[0, 0]
im = ax.imshow(
    real_pdf,
    extent = [PLOT_XLIM[0], PLOT_XLIM[1], PLOT_YLIM[0], PLOT_YLIM[1]],
    origin = 'lower',
    cmap = 'Blues',
    alpha = 0.9
)
ax.set_title('True density', fontsize = TITLE_SIZE)
ax.set_xlim(PLOT_XLIM)
ax.set_ylim(PLOT_YLIM)
ax.set_aspect('equal')

cb = fig.colorbar(im, ax = ax, fraction = 0.04, pad = 0.02)
cb.ax.tick_params(labelsize = LABEL_SIZE)

# Real sample
ax = axes[0, 1]
ax.scatter(
    real_data[:, 0],
    real_data[:, 1],
    s = 5,
    alpha = 0.6
)
ax.set_title('Real samples', fontsize = TITLE_SIZE)
ax.set_xlim(PLOT_XLIM)
ax.set_ylim(PLOT_YLIM)
ax.set_aspect('equal')

# Generated samples
titles = list(METHODS.keys())
positions = [(0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]

for (i, (row, col)) in enumerate(positions):
    data = gen_data_list[i]
    ax = axes[row, col]
    ax.scatter(data[:, 0], data[:, 1], s = 5, alpha = 0.6)
    ax.set_title(titles[i], fontsize = TITLE_SIZE)
    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)
    ax.set_aspect('equal')

# Global tick size
for ax_row in axes:
    for ax in ax_row:
        if ax is not None:
            ax.tick_params(labelsize = LABEL_SIZE)

# plt.tight_layout()
plt.savefig(os.path.join(CONFIG_ID, '2dim_visualization.png'))
plt.savefig(os.path.join(CONFIG_ID, '2dim_visualization.pdf'), format = 'pdf')
plt.close()