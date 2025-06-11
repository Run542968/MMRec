#!/bin/bash
# A PID parameter is not entered, the script does not wait and runs the subsequent task; if a PID parameter is entered, it monitors the process and continues execution after it finishes

# check if the PID is provided
if [ -z "$1" ]; then
    echo "Without PID param, runnign subsequent task directly..."
else
    # get the PID of target process
    target_pid=$1

    # check if the process exists
    if ! ps -p $target_pid > /dev/null; then
        echo "PID $target_pid not exists or unvalid!"
        exit 1
    else
        echo "Start monitor the process, which PID: $target_pid ..."

        # Loop check for PID presence
        while ps -p $target_pid > /dev/null; do
            # if process exists, wait 1 second
            sleep 1
        done

        echo "Process PID $target_pid ends, start run subsequent tasks."
    fi
fi

## baby
# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 0 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw00_tdw00' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500

# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw10_1500' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1500
# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10_1500' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1500
# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10_1500' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1500
# python main.py --gpu_id 0 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00_1500' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1500
# python main.py --gpu_id 0 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw00_tdw10_1500' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1500
# python main.py --gpu_id 1 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw00_1500' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1500
# python main.py --gpu_id 2 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw00_1500' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1500


# python main.py --gpu_id 1 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10_tk25' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 25 30 35 40 --behavior_knn_k 2500
# python main.py --gpu_id 5 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw01' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.1 --text_knn_k 5 10 15 20 25 30 35 40 --behavior_knn_k 2500
# python main.py --gpu_id 2 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw03' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.3 --text_knn_k 5 10 15 20 25 30 35 40 --behavior_knn_k 2500
# python main.py --gpu_id 3 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw05' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.5 --text_knn_k 5 10 15 20 25 30 35 40 --behavior_knn_k 2500
# python main.py --gpu_id 0 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw07' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.7 --text_knn_k 5 10 15 20 25 30 35 40 --behavior_knn_k 2500
# python main.py --gpu_id 2 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw08' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.8 --text_knn_k 5 10 15 20 25 30 35 40 --behavior_knn_k 2500

# python main.py --gpu_id 3 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw00_tdw00_vk25' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 25 30 35 40 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 3 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw01_bgdt0_vdw00_tdw00' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.1 --image_knn_k 5 10 15 20 25 30 35 40 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 2 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw03_bgdt0_vdw00_tdw00' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.3 --image_knn_k 5 10 15 20 25 30 35 40 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 3 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw05_bgdt0_vdw00_tdw00' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.5 --image_knn_k 5 10 15 20 25 30 35 40 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 4 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw07_bgdt0_vdw00_tdw00' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.7 --image_knn_k 5 10 15 20 25 30 35 40 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500
# python main.py --gpu_id 4 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw08_bgdt0_vdw00_tdw00' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.8 --image_knn_k 5 10 15 20 25 30 35 40 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500


# python result_collector.py --dataset 'baby' --exp_list YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10 YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw10 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00 YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw10_1500 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10_1500 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10_1500 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00_1500
# python result_collector.py --dataset 'baby' --exp_list YunJian_v52_mge10_kl_bdw10_bgdt0_vdw00_tdw00 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10_tk25 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw01 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw03 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw05 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw07 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw08 YunJian_v52_mge10_kl_bdw10_bgdt0_vdw00_tdw00_vk25 YunJian_v52_mge10_kl_bdw01_bgdt0_vdw00_tdw00 YunJian_v52_mge10_kl_bdw03_bgdt0_vdw00_tdw00 YunJian_v52_mge10_kl_bdw05_bgdt0_vdw00_tdw00 YunJian_v52_mge10_kl_bdw07_bgdt0_vdw00_tdw00 YunJian_v52_mge10_kl_bdw08_bgdt0_vdw00_tdw00

## microlens
# python main.py --gpu_id 5 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1200
# python main.py --gpu_id 6 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1200
# python main.py --gpu_id 1 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1200
# python main.py --gpu_id 7 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1200
# python main.py --gpu_id 0 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1200

# python main.py --gpu_id 5 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw10_800' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 800
# python main.py --gpu_id 1 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10_800' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 800
# python main.py --gpu_id 0 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10_800' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 800
# python main.py --gpu_id 1 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00_800' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 800
# python main.py --gpu_id 2 --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw10_bgdt0_vdw00_tdw00_800' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 800

# python result_collector.py --dataset 'microlens' --exp_list YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10 YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw10 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00 YunJian_v52_mge10_kl_bdw10_bgdt0_vdw10_tdw10_800 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10_800 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10_800 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00_800 YunJian_v52_mge10_kl_bdw10_bgdt0_vdw00_tdw00_800


## sports
# python main.py --gpu_id 6 --dataset 'sports' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 4000
python main.py --gpu_id 3 --dataset 'sports' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10_bk1000' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2000
# python main.py --gpu_id 4 --dataset 'sports' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00_bk2000' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 0.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2000
# python main.py --gpu_id 7 --dataset 'sports' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10_bk2000' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 0.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2000
# python main.py --gpu_id 2 --dataset 'sports' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10_bk2000' --mge_weight 1.0 --relation_distillation_func 'KL' --behavior_distillation_weight 0.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2000

python result_collector.py --dataset 'sports' --exp_list YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10 YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10_bk1000 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw00_bk2000 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw00_tdw10_bk2000 YunJian_v52_mge10_kl_bdw00_bgdt0_vdw10_tdw10_bk2000





## clothing
python main.py --dataset 'clothing' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1300

## elec
python main.py --dataset 'elec' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 18000

