#!/bin/bash
#SBATCH -A research
#SBATCH --mincpus=40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o op.txt
module load cuda/10.1
module load cudnn/7.6-cuda-10.0
module load TensorRT/7.2.2.3

# Install conda onto /scratch
## Deactivate base
conda deactivate

rm -rf /scratch/ishaanshah/Aman/
mkdir -p /scratch/ishaanshah/Aman/
cd /scratch/ishaanshah
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh
bash Miniconda3-py37_4.11.0-Linux-x86_64.sh -b -p ./miniconda
cd -

echo "Installed Miniconda on scratch"

# Activate conda base installation
source /scratch/ishaanshah/miniconda/etc/profile.d/conda.sh
conda activate

# Manually install all the libraries
python3 -m pip install torch==1.9.0 torchvision==0.10.0 > /dev/null
python3 -m pip install gin-config matplotlib tensorboardX --no-cache-dir > /dev/null
# For some reason this takes time
python3 -m pip install "neuralnet-pytorch[gin] @ git+https://github.com/justanhduc/neuralnet-pytorch.git@6bda19fdc57f176cb82f58d287602f4ccf4cfc23" --no-cache-dir > /dev/null
python3 -m pip install "neuralnet-pytorch[gin] @ git+https://github.com/justanhduc/neuralnet-pytorch.git@6bda19fdc57f176cb82f58d287602f4ccf4cfc23" --global-option="cuda-ext" --no-cache-dir > /dev/null
python3 -m pip install --upgrade timm

echo "Dependencies installed" 

## Now change the third line of the neuralnet_pytorch library
sed -i '3s/.*/from collections import abc as container_abcs/' /scratch/ishaanshah/miniconda/lib/python3.7/site-packages/torch/_six.py

echo "Replaced line 3 for deprecated library"

echo "Copying Dataset"
# Copy dataset 
rsync ishaanshah@ada.iiit.ac.in:/share3/ishaanshah/data.zip /scratch/ishaanshah/
echo "Copied Dataset"

echo "Unzipping Dataset"
# Unzip the dataset
rm -rf /scratch/ishaanshah/Aman/data
mkdir -p /scratch/ishaanshah/Aman/data
unzip /scratch/ishaanshah/data.zip -d /scratch/ishaanshah/Aman/data > /dev/null
echo "Unzipped Dataset"

echo "Moving src to /scratch and unzipping the files"
# Move the source files in /scratch
rsync -r ishaanshah@ada.iiit.ac.in:/share3/ishaanshah/trained_op.zip /scratch/ishaanshah/
cd /scratch/ishaanshah
rm -rf Aman/graphx-conv
unzip trained_op.zip 
rm Aman/training_output.txt
cd -

# Set the configuration folder properly


echo "Testing started.."
# Run the tester 
cd /scratch/ishaanshah/Aman/graphx-conv/src/
python3 ./test.py ./configs/lowrankgraphx-up-final.gin --gpu 4 | tee /scratch/ishaanshah/testing_output.txt
cd -

echo "Testing ended.."

# zip the results directory again
cd /scratch/ishaanshah/
zip -r tested_op.zip Aman/graphx-conv testing_output.txt > /dev/null 
cd -

echo "Zipped files after tesing successfully."
# copy to the /share3 directory 
rsync /scratch/ishaanshah/tested_op.zip ishaanshah@ada.iiit.ac.in:/share3/ishaanshah 
echo "Copied the files to /share3 directory"
