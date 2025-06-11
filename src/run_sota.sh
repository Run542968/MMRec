conda deactivate
source ~/anaconda3_2024/bin/activate
conda activate MMRec
cd ~/MMRec/src


# BPR
python main.py --dataset 'baby' --model 'BPR' --exp_name 'reproduce' --gpu_id 3
python main.py --dataset 'microlens' --model 'BPR' --exp_name 'reproduce' --gpu_id 4
python main.py --dataset 'sports' --model 'BPR' --exp_name 'reproduce' --gpu_id 3

# MG
python main.py --dataset 'baby' --model 'FREEDOM' --exp_name 'reproduce_MG' --mg --gpu_id 2
python main.py --dataset 'microlens' --model 'FREEDOM' --exp_name 'reproduce_MG' --mg --gpu_id 2
python main.py --dataset 'sports' --model 'FREEDOM' --exp_name 'reproduce_MG' --mg --gpu_id 4


python main.py --dataset 'baby' --model 'MGCN' --exp_name 'reproduce_MGCN' --gpu_id 0
python main.py --dataset 'baby' --model 'LGMRec' --exp_name 'reproduce_LGMRec' --gpu_id 1
