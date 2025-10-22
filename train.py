import argparse
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load, YAML  # 
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset, vessel_dataset_train
from trainer import Trainer
from utils import losses
import torch
from pathlib import Path
import os


def main(CFG, data_path, batch_size, with_val=False, save_tag=''):
    from utils.helpers import get_instance, seed_torch
    seed_torch()
    train_dataset = vessel_dataset_train(data_path, mode="training")
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    logger.info('The patch number of train is %d' % len(train_dataset))
    model = get_instance(models, 'model', CFG)
    logger.info(f'\n{model}\n')
    loss = get_instance(losses, 'loss', CFG)
    
    _tune_from = CFG.model.get('fine_tune_from')
    if (_tune_from is not None) and (_tune_from != "") and (_tune_from != "none"):
        checkpoint = torch.load(_tune_from)
        _state_dict = {k.replace('module.', '', 1):v for k,v in checkpoint['state_dict'].items()}
        
        # for key in ['catconv.weight', 'catconv.bias']:
        #     if key in _state_dict:
        #         _state_dict.pop(key)
        model.load_state_dict(_state_dict, strict=CFG.model.get('strict'))
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        # val_loader=val_loader if with_val else None
        val_loader= None,
        save_tag=save_tag,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="dataset/DRIVE/training_pro/ps_96_stride_6", type=str,
                        help='the path of dataset')
    parser.add_argument('-bs', '--batch_size', default=128, type=int,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=False, default=False, action="store_true")
    # parser.add_argument('-g', '--gpu', default='1', type=str)
    parser.add_argument('-y', '--yaml', default='config_simple_rep_resnet.yaml', type=str)
    parser.add_argument('-ch', '--ch', default=16, type=int)
    parser.add_argument('-l', '--layer', default=32, type=int)
    parser.add_argument('-k', '--kernel', default=7, type=int)
    parser.add_argument('--fine_tune_from', default='', type=str)
    args = parser.parse_args()
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.yaml, encoding='utf-8') as file:
        CFG = Bunch(YAML().load(file))
    
    CFG.model['args']['param_list'] = [args.ch, args.layer, args.kernel]
    CFG.model['fine_tune_from'] = args.fine_tune_from
    save_tag = f"ch{args.ch}_layer{args.layer}_kernel{args.kernel}"
    main(CFG, args.dataset_path, args.batch_size, args.val, save_tag)
