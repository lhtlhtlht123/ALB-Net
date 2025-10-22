import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import safe_load, YAML
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files
from tqdm import tqdm
from pathlib import Path

def steel(data_path, name, patch_size, stride, mode):
    root = Path(data_path)
    list_path_input = sorted([root / 'tyre_l' / file for file in os.listdir(root / 'tyre_l') if file.endswith('.jpg')])
    list_path_gt = sorted([root / 'tyre_revise' / file for file in os.listdir(root / 'tyre_revise') if file.endswith('.png')])
    img_list = []
    gt_list = []
    for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt))):
        img = Image.open(str(path_input))
        gt = Image.open(str(path_gt))
        img_list.append(ToTensor()(Grayscale(1)(img)))
        # 将标签转换为二值图，假设阈值是100
        gt_numpy = np.where(np.array(gt) >= 100, 255, 0).astype(np.uint8)
        gt_torch = torch.from_numpy(gt_numpy)[None]
        gt_list.append(gt_torch)
    img_list = normalization(img_list)
    save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
    save_dir.mkdir(exist_ok=True, parents=True)
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
    np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)

def xcad(data_path, name, patch_size, stride, mode):
    root = Path(data_path)
    list_path_input = sorted([root / 'train/img' / file for file in os.listdir(root / 'train/img') if file.endswith('.png')])
    list_path_gt = sorted([root / 'train/label' / file for file in os.listdir(root / 'train/label') if file.endswith('.png')])
    img_list = []
    gt_list = []
    for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt))):
        img = Image.open(str(path_input))
        gt = Image.open(str(path_gt))
        img_list.append(ToTensor()(img))

        # # 取值0/255, 没必要使用float32 存,
        # 但是后续操作是在torch中的
        gt_torch = torch.from_numpy(np.array(gt)[None])
        gt_list.append(gt_torch)
    img_list = normalization(img_list)
    save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
    save_dir.mkdir(exist_ok=True, parents=True)
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
    np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)

def crack(data_path, name, patch_size, stride, mode):
    root = Path(data_path)
    if mode.startswith('train'):
        start, end = 6192, 6636  # STARE训练集的图像编号范围
    elif mode == 'test':
        start, end = 6637, 6780  # STARE测试集的图像编号范围
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    list_path_input = [f for f in root.glob('images/*.jpg') if start <= int(f.stem) <= end]
    list_path_gt = [f for f in root.glob('annotation_mask/*.png') if start <= int(f.stem) <= end]
    assert len(list_path_input) == len(list_path_gt), "Image and GT files count mismatch"
    img_list = []
    gt_list = []

    for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt)), total=len(list_path_input)):
        img = Image.open(str(path_input))
        gt = Image.open(str(path_gt))
        img_list.append(ToTensor()(Grayscale(1)(img)))

        # 取值0/255, 没必要使用float32 存, 但是后续操作是在torch中的
        gt_torch = torch.from_numpy(np.array(gt)[None])
        gt_list.append(gt_torch)

    img_list = normalization(img_list)
    save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
    save_dir.mkdir(exist_ok=True, parents=True)
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
    np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)

def dca1(data_path, name, patch_size, stride, mode):
    root = Path(data_path)
    if mode.startswith('train'):
        start, end = 1, 101
    elif mode == 'test':
        start, end = 101, 135
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    list_path_input = [root / f"{idx}.pgm" for idx in range(start, end)]
    list_path_gt = [root / f"{idx}_gt.pgm" for idx in range(start, end)]
    img_list = []
    gt_list = []

    for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt)), total=len(list_path_input)):
        img = Image.open(str(path_input))
        gt = Image.open(str(path_gt))
        img_list.append(ToTensor()(Grayscale(1)(img)))
        # 将标签转换为二值图，假设阈值是100
        gt_numpy = np.where(np.array(gt) >= 100, 255, 0).astype(np.uint8)
        gt_torch = torch.from_numpy(gt_numpy)[None]
        gt_list.append(gt_torch)

    img_list = normalization(img_list)
    save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
    save_dir.mkdir(exist_ok=True, parents=True)
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
    np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)


def chuac(data_path, name, patch_size, stride, mode):
    root = Path(data_path)
    if mode.startswith('train'):
        start, end = 1, 21
    elif mode == 'test':
        start, end = 21, 31
    else:
        raise ValueError
    list_path_input = [root / 'Original' / f"{idx:02}.png" for idx in range(start, end)]
    list_path_gt = [root / 'Photoshop' / f"angio{idx:02}ok.png" for idx in range(start, end)]
    img_list = []
    gt_list = []
    for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt))):
        img = Image.open(str(path_input)).convert('L') # 可能不需要
        gt = Image.open(str(path_gt)).convert('L')

        img = img.resize((512, 512), Image.LANCZOS)
        img_list.append(ToTensor()(Grayscale(1)(img)))

        # # 取值0/255, 没必要使用float32 存,
        # 但是后续操作是在torch中的
        # gt = np.array(gt)[None]
        gt_torch = torch.from_numpy(np.array(gt)[None])
        # 假设 gt_torch 是你的 PyTorch 张量
        gt_numpy = gt_torch.data.cpu().numpy()

        # 使用布尔索引找出数组中大于0且小于255的元素

        gt_numpy = np.where(gt_numpy >= 100, 255, 0).astype(np.uint8)

        gt_torch = torch.from_numpy(gt_numpy)
        gt_list.append(gt_torch)
    img_list = normalization(img_list)
    save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
    save_dir.mkdir(exist_ok=True, parents=True)
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
    np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)


def chase(data_path, name, patch_size, stride, mode):
    root = Path(data_path)
    if mode.startswith('train'):
        start, end = 1, 21
    elif mode == 'test':
        start, end = 21, 26
    else:
        raise ValueError
    list_path_input = [root / 'train/images' / f"Image_{idx:02}.jpg" for idx in range(start, end)]
    list_path_gt = [root / 'train/label' / f"Image_{idx:02}_1stHO.png" for idx in range(start, end)]
    img_list = []
    gt_list = []
    for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt))):
        img = Image.open(str(path_input)) #可能不需要
        gt = Image.open(str(path_gt)).convert('L')
        img_list.append(ToTensor()(Grayscale(1)(img)))

        # # 取值0/255, 没必要使用float32 存,
        # 但是后续操作是在torch中的
        gt_torch = torch.from_numpy(np.array(gt)[None])
        gt_list.append(gt_torch)
    img_list = normalization(img_list)
    save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
    save_dir.mkdir(exist_ok=True, parents=True)
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
    np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)

def stare(data_path, name, patch_size, stride, mode):
    root = Path(data_path)
    if mode.startswith('train'):
        start, end = 1, 16  # STARE训练集的图像编号范围
    elif mode == 'test':
        start, end = 16, 21  # STARE测试集的图像编号范围
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    list_path_input = [root / 'stare-images' / f"im{idx:04}.ppm" for idx in range(start, end)]
    list_path_gt = [root / 'labels-ah' / f"im{idx:04}.ah.ppm" for idx in range(start, end)]
    img_list = []
    gt_list = []

    for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt)), total=len(list_path_input)):
        img = Image.open(str(path_input))
        gt = Image.open(str(path_gt))
        img_list.append(ToTensor()(Grayscale(1)(img)))

        # 取值0/255, 没必要使用float32 存, 但是后续操作是在torch中的
        gt_torch = torch.from_numpy(np.array(gt)[None])
        gt_list.append(gt_torch)

    img_list = normalization(img_list)
    save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
    save_dir.mkdir(exist_ok=True, parents=True)
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
    np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)

# import glob
# import numpy as np
# from torch.utils.data import Dataset
# import cv2
# import torchvision.transforms as transforms
# from skimage import morphology
# from skan import Skeleton
# from skan import csr
# import random
# from skimage import morphology
# from PIL import Image
#
#
# def drive(data_path, name, patch_size, stride, mode):
#     root = Path(data_path)
#     if mode.startswith('train'):
#         start, end = 21, 41
#     elif mode == 'test':
#         start, end = 1, 21
#     else:
#         raise ValueError
#     list_path_input = [root / 'train/images' / f"{idx:02}_training.tif" for idx in range(start, end)]
#     list_path_gt = [root / 'train/labels' / f"{idx:02}_manual1.png" for idx in range(start, end)]
#     img_list = []
#     gt_list = []
#     for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt))):
#         img = Image.open(str(path_input))
#         gt = Image.open(str(path_gt))
#         # 应用Fundus数据集的处理策略
#         img = np.array(img.resize([512, 512]))
#         gt = np.array(gt.resize([512, 512]))
#         # 应用掩码和噪声
#         if name == "Fundus":
#             sk = morphology.thin(gt).astype(np.uint8)
#             sk_copy = sk.copy()
#             graph_class = Skeleton(sk)
#             branch_data = csr.summarize(graph_class)
#             length = branch_data.shape[0]
#             del_list = random.sample(range(0, length), int(0.1 * length))
#             for del_index in del_list:
#                 integer_coords = tuple(graph_class.path_coordinates(del_index)[1:-1].T.astype(int))
#                 sk[integer_coords] = False
#             sk_del = sk_copy - sk
#             kernel = np.ones((7, 7), np.uint8)
#             masked = cv2.dilate(sk_del, kernel, iterations=1) * 255
#             gauss_r = np.random.normal(0, 20, (gt.shape[0], gt.shape[1]))
#             gauss_r = np.where(masked == 255, gauss_r, masked)
#             gauss_g = np.random.normal(0, 20, (gt.shape[0], gt.shape[1]))
#             gauss_g = np.where(masked == 255, gauss_g, masked)
#             gauss_b = np.random.normal(0, 20, (gt.shape[0], gt.shape[1]))
#             gauss_b = np.where(masked == 255, gauss_b, masked)
#             mask_rgb = np.zeros((masked.shape[0], masked.shape[1], 3))
#             gauss_rgb = np.zeros((masked.shape[0], masked.shape[1], 3))
#             mask_rgb[:, :, 0] = masked
#             mask_rgb[:, :, 1] = masked
#             mask_rgb[:, :, 2] = masked
#             gauss_rgb[:, :, 0] = gauss_r
#             gauss_rgb[:, :, 1] = gauss_g
#             gauss_rgb[:, :, 2] = gauss_b
#             result = np.uint8(np.add(255 - img, mask_rgb))
#             result = np.uint8(np.add(result, gauss_rgb))
#             img = 255 - result
#         # 转换为PyTorch张量
#         img_list.append(ToTensor()(img))
#         gt_list.append(ToTensor()(gt))
#     img_list = normalization(img_list)
#     save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
#     save_dir.mkdir(exist_ok=True, parents=True)
#     img_patch = get_patch(img_list, patch_size, stride)
#     gt_patch = get_patch(gt_list, patch_size, stride)
#     np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
#     np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)

def drive(data_path, name, patch_size, stride, mode):
    root = Path(data_path)
    if mode.startswith('train'):
        start, end = 21, 41
    elif mode == 'test':
        start, end = 1, 21
    else:
        raise ValueError
    list_path_input = [root / 'train/images' / f"{idx:02}_training.tif" for idx in range(start, end)]
    list_path_gt = [root / 'train/labels' / f"{idx:02}_manual1.png" for idx in range(start, end)]
    img_list = []
    gt_list = []
    for i, (path_input, path_gt) in tqdm(enumerate(zip(list_path_input, list_path_gt))):
        img = Image.open(str(path_input))
        gt = Image.open(str(path_gt))
        img_list.append(ToTensor()(Grayscale(1)(img)))

        # # 取值0/255, 没必要使用float32 存,
        # 但是后续操作是在torch中的
        gt_torch = torch.from_numpy(np.array(gt)[None])
        gt_list.append(gt_torch)
    img_list = normalization(img_list)
    save_dir = root / f'{mode}_pro/ps_{patch_size}_stride_{stride}'
    save_dir.mkdir(exist_ok=True, parents=True)
    img_patch = get_patch(img_list, patch_size, stride)
    gt_patch = get_patch(gt_list, patch_size, stride)
    np.save(f'{save_dir}/input_{img_patch.shape}.npy', img_patch)
    np.save(f'{save_dir}/gt_{gt_patch.shape}.npy', gt_patch)

def data_process(data_path, name, patch_size, stride, mode):
    save_path = os.path.join(data_path, f"{mode}_pro")
    if mode.startswith('train'):
        save_path = Path(save_path) / f'ps_{patch_size}_stride_{stride}'
    # dir_exists(save_path)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    remove_files(save_path)
    if name == "DRIVE":
        _dir = 'train' if mode.startswith('train') else mode
        img_path = os.path.join(data_path, _dir, "images")
        gt_path = os.path.join(data_path, _dir, "labels")
        if mode.startswith('train'):
            start, end, tag = 21, 41, 'training'
        elif mode == 'test':
            start, end, tag = 1, 21, 'test'
        else:
            raise ValueError
        file_list = list([f"{x:02}_{tag}.tif" for x in range(start, end)])
    elif name == "CHASEDB1":
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "STARE":
        img_path = os.path.join(data_path, "stare-images")
        gt_path = os.path.join(data_path, "labels-ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "steel":
        data_path = os.path.join(data_path, "tyre_l")
        gt_path = os.path.join(data_path, "tyre_revise")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    img_list = []
    gt_list = []
    for i, file in enumerate(tqdm(file_list)):
        if name == "DRIVE":
            img = Image.open(os.path.join(img_path, file))
            # gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.png"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
        elif name == "steel":
            base_name = os.path.splitext(file)[0]  # 去掉后缀
            img = Image.open(os.path.join(img_path, base_name + ".jpg"))
            gt = Image.open(os.path.join(gt_path, base_name + ".png"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
        elif name == "CHASEDB1":
            if len(file) == 13:
                if mode == "training" and int(file[6:8]) <= 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[6:8]) > 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "DCA1":
            if len(file) <= 7:
                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "CHUAC":
            if mode == "training" and int(file[:-4]) <= 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                    tail = "PNG"
                else:
                    tail = "png"
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[:-4]) > 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok.png"), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
                cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
    img_list = normalization(img_list)
    if mode == "training":
        img_patch = get_patch(img_list, patch_size, stride)
        gt_patch = get_patch(gt_list, patch_size, stride)
        save_patch(img_patch, save_path, f"img_patch", name)
        save_patch(gt_patch, save_path, f"gt_patch", name)
    elif mode == "test":
        if name != "CHUAC":
            img_list = get_square(img_list, name)
            gt_list = get_square(gt_list, name)
        save_each_image(img_list, save_path, "img", name)
        save_each_image(gt_list, save_path, "gt", name)


def get_square(img_list, name):
    img_s = []
    if name == "DRIVE":
        shape = 592
    elif name == "CHASEDB1":
        shape = 1008
    elif name == "DCA1":
        shape = 320
    elif name == "steel":
        shape = 1024
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)
    for i in range(len(img_list)):
        img = pad(img_list[i])
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape  # DRIVE 584, 565
    pad_h = stride - (h - patch_size) % stride  # 4
    pad_w = stride - (w - patch_size) % stride  # 5
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)  # 588, 570
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)  # 88, 1, 91, 48, 48
        image = image.contiguous().view(
            image.shape[0] * image.shape[1] * image.shape[2], patch_size, patch_size) # 88*91, 48, 48
        # for sub2 in image:
        image_list.append(image.contiguous().numpy())
    np_oup = np.concatenate(image_list, axis=0)
    return np_oup


def save_patch(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def save_each_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def normalization(imgs_list):
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="dataset/CHASE", type=str,
                        help='the path of dataset',required=False)
    parser.add_argument('-dn', '--dataset_name', default="CHASEDB1", type=str,
                        help='the name of dataset',choices=['DRIVE','CHASEDB1','STARE','CHUAC','DCA1','CRACK','XCAD','steel',"Fundus"],required=False)
    parser.add_argument('-ps', '--patch_size', default=96,
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=24,
                        help='the stride of image partition')
    args = parser.parse_args()

    # drive(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")

    # stare(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")

    chase(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")

    # chuac(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")

    # dca1(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")

    # crack(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")

    # xcad(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")

    # steel(args.dataset_path, args.dataset_name, args.patch_size, args.stride, "training")


