# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--exp_name', type=str, default='', help='name of experiment.')
    parser.add_argument('--relation_distillation_func', type=str, default='CE', choices=('CE','KL'))
    parser.add_argument('--behavior_distillation_weight', type=float, default=1.0)
    parser.add_argument('--visual_distillation_weight', type=float, default=1.0)
    parser.add_argument('--textual_distillation_weight', type=float, default=1.0)
    parser.add_argument('--mge_weight', type=float, default=1.0)
    parser.add_argument('--behavior_graph_dropout_threshold', type=int, default=0)
    parser.add_argument('--image_knn_k', type=int, default=[5,10,15,20], nargs='+')
    parser.add_argument('--text_knn_k', type=int, default=[5,10,15,20], nargs='+')
    parser.add_argument('--behavior_knn_k', type=int, default=3000)
    parser.add_argument('--train_batch_size', type=int, default=2048)
    
    args, _ = parser.parse_known_args()
    assert _== [], (f"_: {_}")

    config_dict = {
        'gpu_id': args.gpu_id,
        'exp_name': args.exp_name,
        'relation_distillation_func': args.relation_distillation_func,
        'behavior_distillation_weight': args.behavior_distillation_weight,
        'visual_distillation_weight': args.visual_distillation_weight,
        'textual_distillation_weight': args.textual_distillation_weight,
        'mge_weight': args.mge_weight,
        'behavior_graph_dropout_threshold': args.behavior_graph_dropout_threshold,
        'image_knn_k': args.image_knn_k,
        'text_knn_k': args.text_knn_k,
        'behavior_knn_k': args.behavior_knn_k,
        'train_batch_size': args.train_batch_size
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


