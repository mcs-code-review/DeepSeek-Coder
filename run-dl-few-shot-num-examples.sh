#!/bin/bash
# Partition for the job:
#SBATCH --partition=deeplearn

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="dl-few-shot-num-examples"

# The project ID which this job should run under:
#SBATCH --account="punim2247"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Number of GPUs requested per node:
#SBATCH --gres=gpu:4
# Slurm QoS:
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5

# Requested memory per node:
#SBATCH --mem=128G

# Use this email address:
#SBATCH --mail-user=mukhammad.karimov@student.unimelb.edu.au

# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-24:0:00

# Standard output and error log
#SBATCH -o logs/dl-few-shot-num-examples-%N.%j.out # STDOUT
#SBATCH -e logs/dl-few-shot-num-examples-%N.%j.err # STDERR

# Run the job from the directory where it was launched (default)

# The modules to load:
echo "Current modules:"
echo "$(module list)"
echo "Loading modules..."
module load foss/2022a
module load CUDA/12.2.0
module load NCCL/2.19.4-CUDA-12.2.0
module load UCX-CUDA/1.14.1-CUDA-12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0
module load GCCcore/11.3.0
module load Python/3.10.4
echo "Loaded modules:"
echo "$(module list)"

# The job command(s):
source ~/venvs/deepseekcoder/bin/activate

# export LD_LIBRARY_PATH=~/venvs/deepseekcoder/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn

### CodeReviewer IR ###

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/deepseek-coder-33b-instruct \
    --tokenizer_path ./ckpt/deepseek-coder-33b-instruct \
    --conf_path ../config/few-shot-dl-num-examples/deepseek-coder-33b-instruct-cr-faiss-3-num-1.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/deepseek-coder-33b-instruct \
    --tokenizer_path ./ckpt/deepseek-coder-33b-instruct \
    --conf_path ../config/few-shot-dl-num-examples/deepseek-coder-33b-instruct-cr-faiss-3-num-2.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/deepseek-coder-33b-instruct \
    --tokenizer_path ./ckpt/deepseek-coder-33b-instruct \
    --conf_path ../config/few-shot-dl-num-examples/deepseek-coder-33b-instruct-crn-faiss-3-num-1.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/deepseek-coder-33b-instruct \
    --tokenizer_path ./ckpt/deepseek-coder-33b-instruct \
    --conf_path ../config/few-shot-dl-num-examples/deepseek-coder-33b-instruct-crn-faiss-3-num-2.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/Qwen2.5-72B-Instruct \
    --tokenizer_path ./ckpt/Qwen2.5-72B-Instruct \
    --conf_path ../config/few-shot-dl-num-examples/qwen2.5-72B-instruct-cr-faiss-3-num-1.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/Qwen2.5-72B-Instruct \
    --tokenizer_path ./ckpt/Qwen2.5-72B-Instruct \
    --conf_path ../config/few-shot-dl-num-examples/qwen2.5-72B-instruct-cr-faiss-3-num-2.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/Qwen2.5-72B-Instruct \
    --tokenizer_path ./ckpt/Qwen2.5-72B-Instruct \
    --conf_path ../config/few-shot-dl-num-examples/qwen2.5-72B-instruct-crn-faiss-3-num-1.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/Qwen2.5-72B-Instruct \
    --tokenizer_path ./ckpt/Qwen2.5-72B-Instruct \
    --conf_path ../config/few-shot-dl-num-examples/qwen2.5-72B-instruct-crn-faiss-3-num-2.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/Qwen2.5-Coder-32B-Instruct \
    --tokenizer_path ./ckpt/Qwen2.5-Coder-32B-Instruct \
    --conf_path ../config/few-shot-dl-num-examples/qwen2.5-coder-32B-instruct-cr-faiss-3-num-1.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/Qwen2.5-Coder-32B-Instruct \
    --tokenizer_path ./ckpt/Qwen2.5-Coder-32B-Instruct \
    --conf_path ../config/few-shot-dl-num-examples/qwen2.5-coder-32B-instruct-cr-faiss-3-num-2.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/Qwen2.5-Coder-32B-Instruct \
    --tokenizer_path ./ckpt/Qwen2.5-Coder-32B-Instruct \
    --conf_path ../config/few-shot-dl-num-examples/qwen2.5-coder-32B-instruct-crn-faiss-3-num-1.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

python code_review_instruction_parallel.py \
    --ckpt_dir ./ckpt/Qwen2.5-Coder-32B-Instruct \
    --tokenizer_path ./ckpt/Qwen2.5-Coder-32B-Instruct \
    --conf_path ../config/few-shot-dl-num-examples/qwen2.5-coder-32B-instruct-crn-faiss-3-num-2.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 2048 \
    --tp_size 4 \
    --debug False

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -c -n -s
my-job-stats -a -n -s
nvidia-smi
