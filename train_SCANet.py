#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 1/9/2024 8:27 PM
# @Author  : Yaser
# @Site    : 
# @File    : train_incorrect.py
# @Software: PyCharm
import argparse
import os
import torch

from src.datasets.definition import set_global_definition
from src.datasets.legokps_shape_cond_dataset import LegoKPSDefinition
from src.tu.train_setup import set_seed

import lightning as L
import yaml
from src.datasets.lego_ECA import build_dataloader, LegoECADataset

from src.SCANet.SCANet import build_SCANet
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from src.util.common_utils import get_filename

torch.backends.cudnn.benchmark = True


class SaveCheckPointEveryN(ModelCheckpoint):
    def __init__(self, every_n=5, start_save=40, **kwargs):
        super().__init__(**kwargs)
        self.every_n = every_n
        self.start_save = start_save

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if (trainer.current_epoch + 1) % self.every_n == 0 and (trainer.current_epoch + 1) >= self.start_save:
            monitor_candidates = self._monitor_candidates(trainer)
            filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, self.best_model_path)
            previous, self.best_model_path = self.best_model_path, filepath
            self._save_checkpoint(trainer, filepath)


def train(config_path, opts):
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    model_config = config_data['model']
    train_config = config_data['train']
    assemble_iter_net = build_SCANet(model_config, train_config, opts.checkpoint_path)
    checkpoint_callback = SaveCheckPointEveryN(filename='{epoch}-{val_loss_sum:.2f}-{val_trans_acc:.2f}',
                                               every_n=train_config["every_n"],
                                               start_save=train_config["start_save"])

    best_checkpoint_callback = ModelCheckpoint(filename='best_{epoch}-{val_loss_sum:.2f}', save_top_k=2,
                                               monitor='val_loss_sum', )
    if opts.debug:
        suffix = "debug"
    else:
        if 'suffix' in config_data:
            end_set = (opts.start + opts.limit) if opts.limit is not None else "all"
            suffix = config_data['suffix'] + f"_@{opts.start}-{end_set}" + f"_D@{opts.dataset_name}"
        else:
            suffix = ""

    resume = opts.resume
    cfg_name = f'cfg_{get_filename(config_path)}_{suffix}'

    if resume is not None:
        cfg_name = f"{cfg_name}@continue"
        assert os.path.exists(resume)
    logger = CSVLogger(save_dir=f"logs", version=cfg_name)
    trainer = L.Trainer(max_epochs=train_config['epoch_nums'], enable_model_summary=True,
                        accumulate_grad_batches=train_config['accumulate_grad_batches'],
                        callbacks=[checkpoint_callback, best_checkpoint_callback], logger=logger)

    train_dataloader, test_dataloader = build_dataloader(opts, train_config, logger.log_dir)
    trainer.fit(model=assemble_iter_net, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader,
                ckpt_path=resume)


def test(config_path, opts):
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    test_config = config_data['test']
    model_config = config_data['model']
    train_config = config_data['train']
    assemble_iter_net = build_SCANet(model_config, checkpoint=opts.checkpoint_path, train_cfg=train_config)
    assemble_iter_net.eval()
    _, test_dataloader = build_dataloader(opts, train_config)
    cfg_name = f'cfg_{get_filename(config_path)}_test'

    trainer = L.Trainer(enable_model_summary=True, default_root_dir=test_config['out_dir'],
                        logger=CSVLogger(save_dir=f"train_exp_result", version=cfg_name))
    trainer.predict(model=assemble_iter_net, dataloaders=test_dataloader)


def main(opts):
    set_global_definition(LegoKPSDefinition())
    config_path = opts.config_path
    seed = opts.seed
    set_seed(seed)
    is_test = opts.test
    if not is_test:
        train(config_path, opts)
    else:
        test(config_path, opts)


def SCANet_parser(parser):
    parser.add_argument('--config_path', type=str, default='./configs/SCAMet.yaml')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = SCANet_parser(parser)
    parser = LegoECADataset.modify_commandline_options(parser)
    main(parser.parse_args())
