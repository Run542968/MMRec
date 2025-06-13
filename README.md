# MMRec

## Prepare the data
- Due to file size limitations, please download the dataset according to the instructions in `data/README.md`. 

## Prepare the environment
```
conda env create -f environment.yaml -n myenv
conda activate myenv
pip install -r requirements.txt
```
- Please pay attention to the CUDA version.


## Train the model
- The experiment are conducted on NVIDIA GeForce RTX 3090.
```shell
cd ~/MMRec/src

python main.py --gpu_id 0 --dataset 'baby' --model 'DIRD' --exp_name 'TEST' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1500
```
