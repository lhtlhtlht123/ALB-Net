
import torch
from bunch import Bunch
from ruamel.yaml import safe_load, YAML
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
# from tester import Tester
# from utils import losses
from utils.helpers import get_instance
from utils.metrics import get_metrics_f1
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF

# _root = 'saved/Simple'
# yaml_path = 'config_base.yaml'
# txt_tag = 'plain_v2'

yaml_path = 'config_simple_rep.yaml'
_root = 'saved/SimpleRep'
# _root = 'saved/Simple'
# txt_tag = 'plain_rep_v2_tune'
# txt_tag = 'plain_rep_v2_tune2'
# txt_tag = 'plain_rep_v2_tune3'
# txt_tag = 'plain_rep_v2_tune4' # adam + lr 1e-5
# txt_tag = 'plain_rep_v2_tune5' # adamW + lr 1e-5
# txt_tag = 'plain_rep_v2_tune6' # SGD + lr 1e-5

# txt_tag = 'plain_rep_v3'  # ch 1/2/4
txt_tag = 'plain_rep_v3_tune'  # ch 1/2/4


# model_dir_list = sorted(list(Path(_root).glob('ch*')))
# model_dir_list = [str(x) for x in model_dir_list]
# save_dir = Path('saved/Simple_pth')
# save_dir.mkdir(exist_ok=True, parents=True)

def get_model_dir_list():
    model_dir_list = []
    # list_ch = [8, 16]  # v2
    list_ch = [1, 2, 4]  # v3
    list_layer = list(range(12, 20))
    list_k = [7, 9]
    
    for ch in list_ch:
        for layer in list_layer:
            for k in list_k:
                _model_dir = list(Path(_root).glob(f'ch{ch}_layer{layer}_kernel{k}*'))
                # assert len(_model_dir), _model_dir
                _model_dir = sorted(_model_dir, key=lambda x: int(str(x).split('_')[-1]))
                model_dir_list.append(_model_dir[-1])
    return model_dir_list

model_dir_list = get_model_dir_list()


class BatchTest():
    def __init__(self, ):
        self.img, self.gt, self.dataset_path = self.get_test_img_gt()

    def load_cfg(self, model_dir, _yaml_path):
        # yaml_path = 'config_base.yaml'
        with open(_yaml_path, encoding="utf-8") as file:
            # CFG = Bunch(safe_load(file)) # removed
            yaml = YAML(typ='safe', pure=True).load(file)
            CFG = Bunch(yaml)
        name = Path(model_dir).name
        params = name.split('_')
        ch = int(params[0].replace('ch', ''))
        layer = int(params[1].replace('layer', ''))
        kernel = int(params[2].replace('kernel', ''))
        CFG.model['args']['param_list'] = [ch, layer, kernel]
        return CFG

    def get_test_img_gt(self):
        data_path = "dataset/DRIVE"
        test_dataset = vessel_dataset(data_path, mode="test")
        test_loader = DataLoader(test_dataset, 20, shuffle=False,  num_workers=4, pin_memory=False)

        # return test_loader, data_path
        _list_img = []
        _list_gt = []
        for (img, gt) in test_loader:
            _list_img.append(img)
            _list_gt.append(gt)
        list_img = torch.cat(_list_img, dim=0).cuda()
        list_gt = torch.cat(_list_gt, dim=0).cuda()
        return list_img, list_gt, data_path


    def infer_once_weight(self, weight_path):
        model_dir = str(Path(weight_path).parent)
        CFG = self.load_cfg(model_dir, yaml_path)

        checkpoint = torch.load(weight_path)
        model = get_instance(models, 'model', CFG)
        _state_dict = {k.replace('module.', '', 1):v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(_state_dict)
        model.eval()
        model = model.cuda()
        
        _list_f1 = []
        with torch.no_grad():
            pred = model(self.img)
            if self.dataset_path.endswith("DRIVE"):
                H, W = 584, 565
            elif self.dataset_path.endswith("CHASEDB1"):
                H, W = 960, 999
            elif self.dataset_path.endswith("DCA1"):
                H, W = 300, 300

            if not self.dataset_path.endswith("CHUAC"):
                # img = TF.crop(self.img, 0, 0, H, W)
                gt = TF.crop(self.gt, 0, 0, H, W)  # 20, 1, 592, 592 -> 20, 1, 584, 565
                pred = TF.crop(pred, 0, 0, H, W)
            for i in range(pred.shape[0]):
                # img = self.img[i,0,...]
                _gt = gt[i,0]
                pre = pred[i,0]

                f1 = get_metrics_f1(pre, _gt, CFG.threshold)
                _list_f1.append(f1)
        mean_f1 = np.array(_list_f1).mean()
        return mean_f1


    def infer_once_dir(self, model_dir):
        model_list = list(Path(model_dir).glob('*pth'))
        model_list = sorted(model_list, key=lambda x: int(x.stem.split('epoch')[1]))

        _idx = -100
        max_f1 = -1
        print(model_dir, len(model_list))
        if len(model_list) == 0:
            _line = f'{Path(model_dir).name},\tmaximum, idx:{_idx}, epoch:{0}, total_model:{len(model_list)}, metric:{max_f1}\n'
            print(_line, end='\t')
            return _line
        for idx, weight_path in enumerate(tqdm(model_list, desc=f'{model_dir}, weight')):
            f1 = self.infer_once_weight(weight_path)
            if f1 > max_f1:
                max_f1 = f1
                _idx = idx
        _line = f'{Path(model_dir).name},\tmaximum, idx:{_idx}, epoch:{_idx+1}, total_model:{len(model_list)}, metric:{max_f1}\n'
        print(_line, end='\t')
        return _line
    
    def infer_all_dir(self, i=0):
        total = 8
        step = len(model_dir_list) // total
        _list = model_dir_list[step*i:step*(i+1)]
        for model_dir in tqdm(_list):
            _line = self.infer_once_dir(model_dir)
            with open(f'saved/f{txt_tag}_speed_{i}.txt', 'a+') as f:
                f.writelines(_line)
                
    def infer_all_dir_once(self):
        for model_dir in tqdm(model_dir_list):
            _line = self.infer_once_dir(model_dir)
            with open(f'saved/{txt_tag}_speed.txt', 'a+') as f:
                f.writelines(_line)
    

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--index', default=0, type=int)
    # args = parser.parse_args()
    batch = BatchTest()
    # batch.infer_all_dir(args.index)
    batch.infer_all_dir_once()

# export CUDA_VISIBLE_DEVICES=0 && p test_batch.py --index 0