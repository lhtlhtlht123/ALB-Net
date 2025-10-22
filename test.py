import argparse
import torch
from bunch import Bunch
from ruamel.yaml import safe_load, YAML
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance
from utils.metrics import get_metrics
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# model_dir = 'saved/FR_UNet_catconv/240925193458'  # epoch 28 F1 0.831
model_dir_list = sorted(list(Path('saved/Simple').glob('ch*')))
model_dir_list = [str(x) for x in model_dir_list]
save_dir = Path('saved/Simple_pth')
save_dir.mkdir(exist_ok=True, parents=True)


def main_dir_batch():
    for model_dir in tqdm(model_dir_list):
        _line = main_dir(model_dir)
        with open('saved/plain.txt', 'a+') as f:
            f.writelines(_line)

def load_cfg(model_dir):
    # yaml_path = 'config_base.yaml'
    yaml_path = 'config_simple_rep_resnet.yaml'
    with open(yaml_path, encoding="utf-8") as file:
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

def get_test_loader():
    data_path = "dataset/DRIVE"
    # data_path = "dataset/DCA1"
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1, shuffle=False,  num_workers=16, pin_memory=True)

    return test_loader, data_path

def test_once():
    # weight_path = 'saved/Simple/ch16_layer10_kernel7_240930112409/checkpoint-epoch19.pth'
    weight_path = 'saved/SimpleRepResAdc/ch16_layer32_kernel7_250330220312/checkpoint-epoch23.pth'
    # weight_path = 'saved/SimpleRepResAdc/ch1_layer24_kernel7_250402174703/checkpoint-epoch35.pth'
    model_dir = str(Path(weight_path).parent)
    CFG = load_cfg(model_dir)

    for ensemble in [True]:
        CFG.tta = ensemble

        # checkpoint = torch.load(weight_path)
        checkpoint = torch.load(weight_path, map_location=torch.device('cuda:0'))
        CFG_ck = checkpoint['config']
        loss = get_instance(losses, 'loss', CFG_ck)
        model = get_instance(models, 'model', CFG)
        test_loader, data_path = get_test_loader()
        test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, show=True, strict=True)
        single_img_time, pre_tensor, metrics_np = test.test_cache()
        print(ensemble, np.mean(metrics_np, axis=0))

def test_multiple():
    root_dir = Path('saved/SimpleRepResAdc/ch16_layer32_kernel7_250408155047')
    actual_iterations = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90]  # 请根据实际情况修改这个列表
    weight_paths = [
        root_dir / f"checkpoint-epoch{i}.pth" for i in range(11, 41)
        # root_dir / f"checkpoint-epoch1-iter{i}0.pth" for i in range(1, 81)
    ]
    idx = 0
    CFG = load_cfg(root_dir)
    f1_scores_test = []  # 用于存储每个 epoch 的 F1 分数
    inference_times = []  # 用于存储所有图片的推理时间 
    f1_scores_train = [0.6212, 0.7429, 0.7669, 0.7799, 0.7881, 0.7933, 0.7970, 0.7998, 0.8020, 0.8036, 0.8052, 0.8064, 0.8075, 0.8086, 0.8094, 0.8101, 0.8104, 0.8107, 0.8112, 0.8114, 0.8115, 0.8117, 0.8121, 0.8125, 0.8127, 0.8129, 0.8128, 0.8132, 0.8137, 0.8138, 0.8140, 0.8141, 0.8141, 0.8142, 0.8143, 0.8143, 0.8143, 0.8144, 0.8143, 0.8143]
    for weight_path in weight_paths:
        idx+=1

        for ensemble in [True]:
            CFG.tta = ensemble

            checkpoint = torch.load(weight_path)
            CFG_ck = checkpoint['config']
            loss = get_instance(losses, 'loss', CFG_ck)
            model = get_instance(models, 'model', CFG)
            test_loader, data_path = get_test_loader()
            test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, show=False, strict=True)
            single_img_time, pre_tensor, metrics_np = test.test_cache()
            inference_times.append(single_img_time)  # 收集每次推理的时间
            print(idx)
            print(ensemble, np.mean(metrics_np, axis=0))
            f1_scores_test.append(np.mean(metrics_np, axis=0)[1])  # 存储 F1 分数
    # 计算平均推理时间
    mean_inference_time = np.mean(inference_times)
    # 计算标准差
    sample_std_inference_time = np.std(inference_times, ddof=1)
    print(f"Average Inference Time: {mean_inference_time}")
    print(f"Standard Deviation of Inference Time: {sample_std_inference_time:.6f}")
    # epochs = range(1, 41)  # 假设有 40 个 epoch
    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, f1_scores_test, marker='o', linestyle='-', color='b', label='Test')
    # # plt.plot(epochs, f1_scores_train, marker='s', linestyle='--', color='r', label='Weights 2')
    # plt.title('F1 Score Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('F1 Score')
    # plt.legend()  # 添加图例
    # plt.grid(True)
    # # plt.ylim(0.815, 0.826)
    # plt.savefig('f1_score_curve.png')  # 保存图表
    # plt.show()

def test1():
    f1_scores_train = [0.6212, 0.7429, 0.7669, 0.7799, 0.7881, 0.7933, 0.7970, 0.7998, 0.8020, 0.8036, 0.8052, 0.8064,
                       0.8075, 0.8086, 0.8094, 0.8101, 0.8104, 0.8107, 0.8112, 0.8114, 0.8115, 0.8117, 0.8121, 0.8125,
                       0.8127, 0.8129, 0.8128, 0.8132, 0.8137, 0.8138, 0.8140, 0.8141, 0.8141, 0.8142, 0.8143, 0.8143,
                       0.8143, 0.8144, 0.8143, 0.8143]
    plt.figure(figsize=(10, 5))
    plt.plot( f1_scores_train, marker='o', linestyle='-', color='b', label='Train')
    # plt.plot(epochs, f1_scores_train, marker='s', linestyle='--', color='r', label='Weights 2')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()  # 添加图例
    plt.grid(True)
    plt.ylim(0.8, 0.815)
    plt.savefig('f1_score_curve.png')  # 保存图表
    plt.show()
def main_dir(model_dir, flag_save=False):
    print('\n\n', model_dir)
    CFG = load_cfg(model_dir)

    model_list = list(Path(model_dir).glob('*pth'))
    model_list = sorted(model_list, key=lambda x: int(x.stem.split('epoch')[1]))

    test_loader, data_path = get_test_loader()
    model = get_instance(models, 'model', CFG)

    _list_pre = []
    _list_metric = []
    for weight_path in tqdm(model_list):
        checkpoint = torch.load(weight_path)
        CFG_ck = checkpoint['config']
        loss = get_instance(losses, 'loss', CFG_ck)
        test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, False, strict=True)
        pre_tensor, metrics_np = test.test_cache()
        print(Path(weight_path).name, np.mean(metrics_np, axis=0))
        if flag_save:
            _list_pre.append(pre_tensor)
        _list_metric.append(metrics_np)
    if flag_save:
        pre = torch.stack(_list_pre, dim=0)
    metric = np.stack(_list_metric, axis=0)
    sort_metric = np.argsort(np.mean(metric[:, :, 1], axis=1))
    _idx = sort_metric[-1]
    _line = f'{Path(model_dir).name},\tmaximum, idx:{_idx}, epoch:{sort_metric[-1]+1}, metric:{np.mean(metric, axis=1)[_idx]}\n'
    
    if flag_save:
        name = Path(model_dir).name
        _path_gt = f"{save_dir}/{name}_predict_len{len(model_list)}x20x585x565.pth"
        if not Path(_path_gt).exists():
            torch.save(pre, _path_gt)
        _path_pre = f"{save_dir}/{name}_metrics_len{len(model_list)}x20x7.npy"
        if not Path(_path_pre).exists():
            np.save(_path_pre, metric)
    print(_line, end='\t')
    return _line


def get_gt():
    data_path = "dataset/DRIVE"
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1, shuffle=False,  num_workers=16, pin_memory=True)
    gt_list = []
    import torchvision.transforms.functional as TF
    H, W = 584, 565
    for (img, gt) in test_loader:
        gt = TF.crop(gt, 0, 0, H, W)
        gt_list.append(gt.squeeze())
    return torch.stack(gt_list)


def load_pre_metric(model_dir):
    _dir = save_dir
    name = Path(model_dir).name
    _path_pth = list(Path(_dir).glob(f"{name}*pth"))
    assert len(_path_pth) == 1
    _path_pth = _path_pth[0]
    _path_npy = list(Path(_dir).glob(f"{name}*npy"))
    assert len(_path_npy) == 1
    _path_npy = _path_npy[0]
    
    pre = torch.load(_path_pth)
    metric = np.load(_path_npy)
    return pre, metric

def check_dir_batch():
    for model_dir in model_dir_list:
        check_dir(model_dir)

def check_dir(model_dir):  # ensemble
    print('\n\n', model_dir)
    pre, metric = load_pre_metric(model_dir)
    gt = get_gt()
    
    mean_metric = np.mean(metric[:, :, 1], axis=1)
    min_list = np.argsort(mean_metric)
    for i in range(1, 6):
        print(i, min_list[-1 * i], mean_metric[min_list[-1 * i]])
    for i in range(1, 6):
        pred = pre[min_list[-1]]
        for j in range(2, i+1):
            _idx = min_list[-1 * j]
            if pred is None:
                pred = pre[_idx]
            else:
                pred += pre[_idx]
        pred = pred / (i * 1.0)
        print(pred.shape, pred.dtype)
        _metric = get_metrics(pred, gt, 0.5)
        print(i, _metric)


def get_best(model_dir):
    _dir = save_dir
    name = Path(model_dir).name
    _path_pth = list(Path(_dir).glob(f"{name}*pth"))
    assert len(_path_pth) == 1
    _path_pth = _path_pth[0]
    _path_npy = list(Path(_dir).glob(f"{name}*npy"))
    assert len(_path_npy) == 1
    _path_npy = _path_npy[0]
    
    pre = torch.load(_path_pth)
    metric = np.load(_path_npy)
    mean_metric = np.mean(metric[:, :, 1], axis=1)
    min_list = np.argsort(mean_metric)
    return pre[min_list[-1]], mean_metric[min_list[-1]]


def ensemble_best():
    for _dir in model_dir_list:
        pred = None
        pre, m = get_best(model_dir)
        print(_dir, m)
        if pred is None:
            pred = pre
        else:
            pred = pred + pre
    pred = pred / len(_dir_list)
    
    gt = get_gt()
    metric = get_metrics(pred, gt, 0.5)
    print(metric)
    

if __name__ == '__main__':
    # test_once()
    # main_dir(model_dir_list[0])
    # check()

    main_dir_batch()
    # check_dir_batch()
    # ensemble_best()
