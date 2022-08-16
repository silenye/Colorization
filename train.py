import os

os.environ["RANK"] = "0"
os.environ['WORLD_SIZE'] = '1'
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from models.model import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")

log_path = 'train_log'


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 1e-5 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (1e-5 - 1e-6) * mul + 1e-6


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


# def train(model, local_rank):
def train(model):
    # if local_rank == 0:
    writer = SummaryWriter('train')
    writer_val = SummaryWriter('validate')
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    # sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=6, num_workers=8, pin_memory=True, drop_last=True)
    dataset_val = VimeoDataset_valid('validation')
    val_data = DataLoader(dataset_val, batch_size=4, pin_memory=True, num_workers=8, drop_last=True)
    args.step_per_epoch = train_data.__len__()
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        # sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            # data_gpu = data.to(device, non_blocking=True) / 255.
            data_gpu = data.to(device) / 255.
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step)
            pred, info = model.update(imgs, gt, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            # if step % 200 == 1 and local_rank == 0:
            if step % 200 == 1:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                # writer.add_scalar('loss/cons', info['loss_cons'], step)
            # if step % 1000 == 1 and local_rank == 0:
            if step % 1000 == 1:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3,
                #                                                                1).detach().cpu().numpy() * 255).astype(
                #     'uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                # flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(4):
                    imgs = np.concatenate((pred[i], gt[i]), 1)#[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    # writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1),
                    #                  step, dataformats='HWC')
                    # writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            # if local_rank == 0:
            print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch,
                                                                            data_time_interval, train_time_interval,
                                                                            info['loss_l1']))
            step += 1
        nr_eval += 1
        if nr_eval % 1 == 0:
            psnr = evaluate(model, val_data, step, writer_val)
        model.save_model(log_path, psnr)
        # dist.barrier()


# def evaluate(model, val_data, nr_eval, local_rank, writer_val):
def evaluate(model, val_data, nr_eval, writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    # dataset_val = VimeoDataset_valid('validation')
    # val_data = DataLoader(dataset_val, batch_size=32, pin_memory=True, num_workers=8)

    for i, data in enumerate(val_data):
        # data_gpu = data.to(device, non_blocking=True) / 255.
        data_gpu = data.to(device) / 255.
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            # merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        #loss_tea_list.append(info['loss_con'].cpu().numpy())
        # loss_distill_list.append(info['loss_distill'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            # psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            # psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        # flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        # if i == 0 and local_rank == 0:
        if i == 0:
            for j in range(4):
                imgs = np.concatenate((pred[j], gt[j]), 1)#[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                # writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')

    eval_time_interval = time.time() - time_stamp

    # if local_rank != 0:
    #     return
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    # writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
    return np.array(psnr_list).mean()


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
    model.load_model('train_log')
    train(model)

