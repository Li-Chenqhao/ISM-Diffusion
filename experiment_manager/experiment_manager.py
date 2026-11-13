import os
import yaml

# Configuartion id
CONFIG_ID = "baseline"

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
        "num_repetitions": 20,
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
        "alpha": 0.1,
    },
}

# ---------------------------------------------------------
# Method-specific configuration
# ---------------------------------------------------------
METHODS = {
    "ISM_AutoGrad": {
        "tag": "A",
        "method": "autograd",
        "network": "repu",
        "script": "ISM_Diffusion.py",
        "user": "",
    },
    "ISM_Hutchinson": {
        "tag": "H",
        "method": "hutchinson",
        "network": "repu",
        "script": "ISM_Diffusion.py",
        "user": "",
    },
    "ISM_ExactGrad": {
        "tag": "E",
        "method": None,
        "network": "repu",
        "script": "ISM_Diffusion_exactgrad.py",
        "user": "",
    },
    "DSM_ReLU": {
        "tag": "DL",
        "method": None,
        "network": "relu",
        "script": "DSM.py",
        "user": "",
    },
    "DSM_RePU": {
        "tag": "DP",
        "method": None,
        "network": "repu",
        "script": "DSM.py",
        "user": "",
    }
}

# Target distributions
DISTRIBUTIONS = {
    "polygon_gmm": "polygon_gmm",
    "gaussian_pushforward": "gauss_push",
    "energy_model": "energy_model",
}

# ---------------------------------------------------------
# Generate JOB files
# ---------------------------------------------------------
def generate_job_file(method_dir, info, config_id):
    BASE_JOB_NAME = "ismd"
    job_path = os.path.join(method_dir, BASE_JOB_NAME + ".job")

    # Lowercase tag
    tag_lower = info["tag"].lower()
    job_name = f"{BASE_JOB_NAME}_{tag_lower}"

    # Job path and anaconda init path
    work_path = f"/users/{info['user']}/{job_name}"  # e.g. "/users/USERNAME/ismd_a"
    ANACONDA_INIT = "/project/Stat/local/anaconda3-2025.12.sh"
    ENV_NAME = "albireo"

    script = info["script"]

    job_content = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --time=23:59:59
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH -e ismd_%A_%a.err
#SBATCH -o ismd_%A_%a.out

cd {work_path}
source {ANACONDA_INIT}
conda activate {ENV_NAME}

# Map array index to distribution
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    DIST="{DISTRIBUTIONS['polygon_gmm']}"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    DIST="{DISTRIBUTIONS['gaussian_pushforward']}"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    DIST="{DISTRIBUTIONS['energy_model']}"
else
    echo "Invalid task id"
    exit 1
fi

CONFIG_ID={config_id}_{info['tag']}

echo "Running [Job ID]: $SLURM_JOB_ID, [Task ID]: $SLURM_ARRAY_TASK_ID, [Distribution]: $DIST, [Config ID]: $CONFIG_ID"
python3 {script} --distribution "$DIST" --id "$CONFIG_ID"

conda deactivate
"""

    with open(job_path, "w", newline = "\n") as f:
        f.write(job_content)

    print(f"[OK] job file created: {job_path}")

# ---------------------------------------------------------
# General YAML configuration files
# ---------------------------------------------------------
def generate_yaml(method_dir, info, config_id):
    configs_root = os.path.join(method_dir, "configs")

    for dist in DISTRIBUTIONS.values():
        dist_dir = os.path.join(configs_root, dist)
        os.makedirs(dist_dir, exist_ok = True)

        tag = info["tag"]
        yaml_path = os.path.join(dist_dir, f"{config_id}_{tag}.yaml")

        config = BASE_CONFIG.copy()
        config["method"] = info["method"]
        config["net"]["type"] = info["network"]

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, sort_keys = False)

        print(f"[OK] config created: {yaml_path}")

# ---------------------------------------------------------
# Genrate results directories
# ---------------------------------------------------------
'''
def generate_results_dirs(method_dir):
    results_root = os.path.join(method_dir, "results")
    for dist in DISTRIBUTIONS.values():
        os.makedirs(os.path.join(results_root, dist), exist_ok = True)
    print(f"[OK] results dirs created under {results_root}")
'''

# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------
def main(config_id):
    root_dir = config_id

    for method_name, info in METHODS.items():
        # Method directory
        method_dir = os.path.join(root_dir, method_name)

        # configs folder and yaml
        generate_yaml(method_dir, info, config_id)

        # results directories
        # generate_results_dirs(method_dir)

        # job file
        generate_job_file(method_dir, info, config_id)

    print("\n=== ALL DONE ===")
    print(f"Experiment directory created at: {root_dir}")

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    config_id = CONFIG_ID
    main(config_id)