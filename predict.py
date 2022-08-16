import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from models.model import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank')
    parser.add_argument('--world_size', default=-1, type=int, help='world size')
    args = parser.parse_args()
    # torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    # torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = Model(args.local_rank)
    model.load_model('checkpoints')

    reference_img1 = r'test_img/Disney_v4_21_036774_s1/frame1.jpg'
    # reference_img2 = r''

    skecth_img = r'test_img/Disney_v4_21_036774_s1/frame2_edge.jpg'
    gt_img = r'test_img/Disney_v4_21_036774_s1/frame2.jpg'
    Ir_1 = cv2.imread(reference_img1)
    Ir_1 = cv2.cvtColor(Ir_1, cv2.COLOR_BGR2RGB)
    Ir_1 = cv2.resize(Ir_1, (512, 288), interpolation=cv2.INTER_AREA)
    # Ir_2 = cv2.imread(inference_img2)
    # Ir_2 = cv2.cvtColor(Ir_2, cv2.COLOR_BGR2RGB)
    # Ir_2 = cv2.resize(Ir_2, (512, 288), interpolation=cv2.INTER_AREA)
    Is = cv2.imread(skecth_img)
    Is = cv2.cvtColor(Is, cv2.COLOR_BGR2RGB)
    Is = cv2.resize(Is, (512, 288), interpolation=cv2.INTER_AREA)
    gt = cv2.imread(gt_img)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = cv2.resize(gt, (512, 288), interpolation=cv2.INTER_AREA)

    Ir_1 = torch.from_numpy(Ir_1.copy()).permute(2, 0, 1) / 255
    # Ir_2 = torch.from_numpy(Ir_2.copy()).permute(2, 0, 1) / 255
    Is = torch.from_numpy(Is.copy()).permute(2, 0, 1) / 255
    gt = torch.from_numpy(gt.copy()).permute(2, 0, 1) / 255

    Ir_1 = Ir_1.unsqueeze(0).to(device)
    # Ir_2 = Ir_2.unsqueeze(0).to(device)
    Is = Is.unsqueeze(0).to(device)
    model.device()
    # imgs = torch.cat((Ir_1, Ir_2), dim=1)
    result = model.inference(Ir_1, Is).detach().cpu().squeeze()
    result = torch.clamp(result, 0, 1)
    result = result.numpy() * 255
    result = result.astype('uint8')
    result = np.transpose(result, (1, 2, 0))
    cv2.imwrite(r'test_img/Disney_v4_21_036774_s1/output.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
