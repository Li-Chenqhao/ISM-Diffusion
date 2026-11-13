import copy
import os
import yaml

# ---------------------------------------------------------
# Experiemntal configurations
# ---------------------------------------------------------
BASE_CONFIG = {
    # Important params
    "dim": 10,
    "method": None,

    # Experimental setting params
    "experiment": {
        "seed": 42,
        "num_repetitions": 1,
    },
    
    # Diffusion process params
    "diffusion": {
        "beta": 1.0,
        "t_lower": 0.01,
        "t_upper": 1.0,
        "t_factor": 0.5,
    },

    # Network architecture params
    "net": {
        "type": "repu",
        "nd": 3,
        "nw": 128,
        "p": 2,
        "leak": 0.1,
    },

    # Training params
    "train": {
        "lr": 1e-3,
        "num_iters": 20000,
        "bs": 256,
        "step": 5000,
        "gamma": 0.5,
        "log_step": 1000,
    },

    # Backward sampling params
    "sampling": {
        "sample_step": 500,
    },

    # Evaluation params
    "eva": {
        "score_val_size": 512,
        "score_test_size": 20000,
        "w1_size_true": 5000,
        "w1_size_gen": 5000,
        "mmd_size_true": 5000,
        "mmd_size_gen": 5000,
        "w1_reg": 0.05,
        "mmd_bandwidth": None,
    },

    # Target distribution params
    "model": {
        "simple_matrix": False,
        "alpha": 0.1
    }
}

# Grid of dimension
DIM_LIST = [8, 12, 16, 20]

# Network architecture parameters
ND_LIST = [3, 3, 3, 3]
NW_LIST = [128, 160, 192, 224]

# Training parameters
LR_LIST = [1e-3, 1e-3, 2e-3, 4e-3]
NITER_LIST = [20000, 24000, 28000, 32000]

# ---------------------------------------------------------
# General YAML configuration files
# ---------------------------------------------------------
def generate_yaml(config_path):
    for i, dim in enumerate(DIM_LIST):
        yaml_path = f'{config_path}_{dim}.yaml'

        config = copy.deepcopy(BASE_CONFIG)
        config["dim"] = dim
        config["net"]["nd"] = ND_LIST[i]
        config["net"]["nw"] = NW_LIST[i]
        config["train"]["lr"] = LR_LIST[i]
        config["train"]["num_iters"] = NITER_LIST[i]

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, sort_keys = False)

        print(f"[OK] config created: {yaml_path}")

# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------
def main(config_root):
    os.makedirs(config_root, exist_ok = True)
    config_path = f"{config_root}/runtime"
    # Generate configs yaml
    generate_yaml(config_path)

    print("\n=== ALL DONE ===")
    print(f"Config files created at: {config_path}")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    config_root = "configs"
    main(config_root)