#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=34G
#SBATCH --mail-type=END,FAIL
#SBATCH -o op.txt
#SBATCH --job-name=neural_lm
module load cuda/10.1
module load cudnn/7.6-cuda-10.0
module load TensorRT/7.2.2.3
source /home2/zishan.kazi/miniconda3/etc/profile.d/conda.sh
conda activate py37

# Manually install all the libraries
python3 -m pip install torch
python3 -m pip install "neuralnet-pytorch[gin] @ git+https://github.com/justanhduc/neuralnet-pytorch.git@6bda19fdc57f176cb82f58d287602f4ccf4cfc23" --global-option="--cuda-ext"
python3 -m pip install gin-config 
python3 -m pip install matplotlib
python3 -m pip install tensorboardX 

echo "Libraries installed successfully"
