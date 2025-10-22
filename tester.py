import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm

from models.simple_rep_resnet import BlockRepRes
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, count_connect_component
import ttach as tta


class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, dataset_path, show=False, strict=True):
        # super(Trainer, self).__init__()
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader
        # self.model = nn.DataParallel(model.cuda())
        self.model = model.cuda()
        self.dataset_path = dataset_path
        self.show = show
        _state_dict = {k.replace('module.', '', 1):v for k,v in checkpoint['state_dict'].items()}
        # import ipdb;ipdb.set_trace()
        # 修改 state_dict 中的键名
        # new_state_dict = {}
        # for key in _state_dict.keys():
        #     new_key = key.replace("list_", "list_module.")  # 根据实际情况调整替换规则
        #     new_state_dict[new_key] = _state_dict[key]
        #
        # 加载修改后的 state_dict
        # self.model.load_state_dict(new_state_dict, strict=strict)
        self.model.load_state_dict(_state_dict, strict=strict)  # TCL相当于是去掉了norm的统计给手动算了
        if self.show:
            print('show')
            dir_exists("save_picture")
            remove_files("save_picture")
        cudnn.benchmark = True

    def test(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                pre = self.model(img)
                loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)

                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300

                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)
                img = img[0,0,...]
                gt = gt[0,0,...]
                pre = pre[0,0,...]
                if self.show:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                    cv2.imwrite(
                        f"save_picture/img{i}.png", np.uint8(img.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture/pre{i}.png", np.uint8(predict*255))
                    cv2.imwrite(
                        f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))

                if self.CFG.DTI:
                    pre_DTI = double_threshold_iteration(
                        i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(
                        *get_metrics(pre, gt, predict_b=pre_DTI).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    self._metrics_update(
                        *get_metrics(pre, gt, self.CFG.threshold).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(
                            pre, gt, threshold=self.CFG.threshold))
                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        if self.CFG.CCC:
            logger.info(f'     CCC:  {self.CCC.average}')
        _lines = []
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')
            _lines.append(str(v))
        _lines = '|' + '|'.join(_lines)+'|'
        print(_lines)
        return _lines
    
    def test_cache(self):
        self.model.reparameterize()
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        _list_pre = []
        _list_metric = []
        total_time = 0
        idx = 0
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):

                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                torch.cuda.synchronize()  # 同步CUDA操作，确保计时准确
                img_time = time.time()  # 记录开始推理单张图片的时间

                pre = self.model(img)
                torch.cuda.synchronize()  # 同步CUDA操作，确保计时准确
                single_img_time = time.time() - img_time # 模型推理结束，计算推理时间
                # print(single_img_time)   # 打印单张图片推理时间
                if i > 0:
                    total_time+=single_img_time
                    idx+=1
                self.batch_time.update(time.time() - tic)

                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                    # H, W = 512, 512
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("CHASE"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300
                elif self.dataset_path.endswith("STARE"):
                    H, W = 605, 700
                elif self.dataset_path.endswith("CRACK"):
                    H, W = 600, 800
                elif self.dataset_path.endswith("XCAD"):
                    H, W = 512, 512
                elif self.dataset_path.endswith("steel"):
                    H, W = 1024, 1024
                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)
                img = img[0,0,...]
                gt = gt[0,0,...]
                pre = pre[0,0,...]
                if self.show:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                    cv2.imwrite(
                        f"save_picture/img{i}.png", np.uint8(img.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture/pre{i}.png", np.uint8(predict*255))
                    cv2.imwrite(
                        f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))
                metric_list = list(get_metrics(pre, gt, self.CFG.threshold).values())
                _list_pre.append(pre.detach().cpu())
                _list_metric.append(metric_list)
                self._metrics_update(*metric_list)
                # single_img_time = time.time() - img_time # 记录处理单张图片所需的时间
                # print(single_img_time) #打印
                tbar.set_description(
                    'TEST ({}) | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()
        print('单张图片推理时间', total_time/idx)
        single_img_avgtime = total_time/idx
        pre = torch.stack(_list_pre, dim=0)
        metric = np.stack(_list_metric, axis=0)
        return single_img_avgtime, pre, metric
        