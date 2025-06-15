# 时间效率分析
python main.py --gpu_id 2 --dataset 'baby' --model 'YunJian_v53' --exp_name 'TEST_1' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 20 --textual_distillation_weight 1.0 --text_knn_k 15 --behavior_knn_k 1500
python main.py --gpu_id 2 --dataset 'baby' --model 'YunJian_v53' --exp_name 'TEST_group_user_base' --mge_weight 1.0 --behavior_graph_dropout_threshold 0 --image_knn_k 20 --text_knn_k 15 --behavior_knn_k 1500


python main.py --gpu_id 2 --dataset 'baby' --model 'LGMRec' --exp_name 'TEST'
python main.py --gpu_id 2 --dataset 'baby' --model 'FREEDOM' --exp_name 'TEST'
python main.py --gpu_id 2 --dataset 'baby' --model 'BM3' --exp_name 'TEST'