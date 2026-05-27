#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --partition=<your-partition>
#SBATCH --account=<your-account>

# Show node name
hostname

# Source start env
source compute_clusters/start_env.sh

# Execute the command passed as argument
eval "$@"