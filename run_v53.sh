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
python main.py --gpu_id 0 --dataset 'baby' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 2500

## microlens
python main.py --dataset 'microlens' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1200

## sports
python main.py --dataset 'sports' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 4000

## clothing
python main.py --dataset 'clothing' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 1300

## elec
python main.py --dataset 'elec' --model 'YunJian_v53' --exp_name 'YunJian_v52_mge10_ce_bdw10_bgdt0_vdw10_tdw10' --mge_weight 1.0 --relation_distillation_func 'CE' --behavior_distillation_weight 1.0 --behavior_graph_dropout_threshold 0 --visual_distillation_weight 1.0 --image_knn_k 5 10 15 20 --textual_distillation_weight 1.0 --text_knn_k 5 10 15 20 --behavior_knn_k 18000

