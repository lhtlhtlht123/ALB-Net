import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation
import numpy as np
from pathlib import Path


class vessel_dataset_train(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):

        self.mode = mode
        self.data_input = self.get_data(path, 'input')
        self.data_gt = self.get_data(path, 'gt')
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        self.length = len(self.data_input)
        print('length', self.length)
        # import ipdb;ipdb.set_trace()

    def get_data(self, path, tag):
        path_list = list(Path(path).glob(f'{tag}*'))
        assert len(path_list) == 1
        data_path = str(path_list[0])
        data = np.load(data_path)
        data = torch.from_numpy(data.astype(np.float32)[:, None])  # Len, H, W -> Len, 1, H, W
        return data

    def __getitem__(self, idx):
        img = self.data_input[idx]
        gt = self.data_gt[idx].to(torch.float32) / 255.  # 255 -> 1.

        seed = torch.seed()
        torch.manual_seed(seed)
        img = self.transforms(img)
        torch.manual_seed(seed)
        gt = self.transforms(gt)

        return img, gt

    def __len__(self):
        return self.length


class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):
        assert mode == 'test'
        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        self.img_file = self._select_img(self.data_file)
        # if split is not None and mode.startswith("train"):
        #     assert split > 0 and split < 1
        #     if not is_val:
        #         self.img_file = self.img_file[:int(split*len(self.img_file))]
        #     else:
        #         self.img_file = self.img_file[int(split*len(self.img_file)):]
        # self.transforms = Compose([
        #     RandomHorizontalFlip(p=0.5),
        #     RandomVerticalFlip(p=0.5),
        #     Fix_RandomRotation(),
        # ])
        self.img_list, self.gt_list = self.cache_testset()

    def cache_testset(self):
        print('start to cache testset')
        img_list = []
        gt_list = []
        for idx in range(len(self.img_file)):
            img_file = self.img_file[idx]
            path_inp = os.path.join(self.data_path, img_file)
            with open(path_inp, mode='rb') as file:
                img = torch.from_numpy(pickle.load(file)).float()
                img_list.append(img)
            
            gt_file = "gt" + img_file[3:]
            path_gt = os.path.join(self.data_path, gt_file)
            with open(path_gt, 'rb') as file:
                gt = torch.from_numpy(pickle.load(file)).float()
                gt_list.append(gt)
        print('cache dataset done')
        return img_list, gt_list
            
    def __getitem__(self, idx):
        img = self.img_list[idx]
        gt = self.gt_list[idx]
        # img_file = self.img_file[idx]
        # with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
        #     img = torch.from_numpy(pickle.load(file)).float()
        # gt_file = "gt" + img_file[3:]
        # with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
        #     gt = torch.from_numpy(pickle.load(file)).float()

        # if self.mode.startswith("train") and not self.is_val:
        #     seed = torch.seed()
        #     torch.manual_seed(seed)
        #     img = self.transforms(img)
        #     torch.manual_seed(seed)
        #     gt = self.transforms(gt)

        return img, gt

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.img_file)
