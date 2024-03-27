#!/bin/bash
# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="33b-instruct-codereview"

# The project ID which this job should run under:
#SBATCH --account="punim2247"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# Slurm QoS:
##SBATCH --qos=gpgpudeeplearn

# Requested memory per node:
##SBATCH --mem=128G

# Use this email address:
#SBATCH --mail-user=mukhammad.karimov@student.unimelb.edu.au

# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=2-00:0:00

# Standard output and error log
#SBATCH -o logs/33b-instruct-codereview.log

# Run the job from the directory where it was launched (default)

# The modules to load:
module load CUDA/11.7.0
module load UCX-CUDA/1.13.1-CUDA-11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load PyTorch/2.1.2-CUDA-11.7.0

# The job command(s):
source .venv/bin/activate

python code_review_instruction.py \
    --ckpt_dir ./ckpt/deepseek-coder-33b-instruct \
    --tokenizer_path ./ckpt/deepseek-coder-33b-instruct \
    --conf_path ../config/deepseek-coder-33b-instruct-codereview.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 512 \
    --debug False

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
