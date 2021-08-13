#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

import pydevd
pydevd.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        if cfg.INTERPRATABLE.VISUALIZATION:
            DefaultTrainer.visualize(cfg, model)
            return
        else:
            res = DefaultTrainer.test(cfg, model)
            return res

    trainer = DefaultTrainer(cfg)
    if args.finetune: Checkpointer(trainer.model).load(cfg.MODEL.WEIGHTS)  # load trained model to funetune

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


# source activate cuda10.2

# visialization:
# market:
# python tools/train_net.py --config-file ./configs/Market1501/sbs_R50_model2_phoenix.yml --eval-only MODEL.WEIGHTS /home/yang/PycharmProjects/CVPR2021-PersonReID-FaceAlignment/fast-reid-master-baseline-v9/results_kiki/market/sbs_R50/model_final.pth TEST.IMS_PER_BATCH 1 INTERPRATABLE.VISUALIZATION True
# duke:
# python tools/train_net.py --config-file ./configs/DukeMTMC/sbs_R50_model2_phoenix.yml --eval-only MODEL.WEIGHTS /home/yang/PycharmProjects/CVPR2021-PersonReID-FaceAlignment/fast-reid-master-baseline-v9/results_kiki/duke/sbs_R50/model_final.pth TEST.IMS_PER_BATCH 1 INTERPRATABLE.VISUALIZATION True
