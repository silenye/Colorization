import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from models.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from models.Flow_Estimate_Net import *
import torch.nn.functional as F
from models.vgg import *
from models.unet_model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flow_estimate_net = Flow_Estimate_Net()

        degrad_params = list(map(id, self.flow_estimate_net.degrad.parameters()))
        base_params = filter(lambda p: id(p) not in degrad_params, self.flow_estimate_net.parameters())
        params = [
            {"params": base_params, 'name':'flow', "lr": 0, "weight_decay": 1e-3},
            {"params": self.flow_estimate_net.degrad.parameters(), 'name': 'degrad', "lr": 0, "weight_decay":0}
        ]
        self.optimG = AdamW(self.flow_estimate_net.parameters(), lr=0.0001)
        self.device()
        self.vgg = VGGPerceptualLoss().to(device)
        if local_rank != -1:
            self.flow_estimate_net = DDP(self.flow_estimate_net, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flow_estimate_net.train()

    def eval(self):
        self.flow_estimate_net.eval()

    def device(self):
        self.flow_estimate_net.to(device)

    def load_model(self, path):
        self.flow_estimate_net.load_state_dict((torch.load('{}/pre-trained.pth'.format(path))))

    def save_model(self, path, psnr, rank=0):
        if rank == 0:
            torch.save(self.flow_estimate_net.state_dict(), '{}/{}.pth'.format(path, psnr))

    def inference(self, imgs, deg, scale=[4, 2, 1], blend=False):
        self.eval()
        # with torch.no_grad():
        merged, u_merged = self.flow_estimate_net(imgs, deg=deg, scale=scale, blend=blend)
        return u_merged

    def get_flow(self, imgs, deg, scale=[4, 2, 1]):
        self.eval()
        flow_list, mask_list = self.flow_estimate_net(imgs, deg=deg, scale=scale)
        return flow_list, mask_list


    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, sketch=None, blend=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        scale = [4, 2, 1]
        # merged, loss_cons = self.flow_estimate_net(imgs[:, :6], imgs[:, 6:9], gt=gt, scale=scale, training=training, blend=blend)
        merged, u_merged = self.flow_estimate_net(imgs[:, :3], imgs[:, 3:6], gt=gt, scale=scale, training=training, blend=blend)
        alpha = 0.5
        loss_l1 = (merged - gt).abs().mean()# * (1 - alpha)
        loss_l1_u = (u_merged - gt).abs().mean()
        # vgg_loss = self.vgg(merged, gt).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = 0.5*loss_l1 + loss_l1_u# + 0.01*vgg_loss
            # loss_G = loss_l1
            loss_G.backward()
            self.optimG.step()
        return u_merged, {
            'sketch': sketch,
            # 'flow': flow[1][:, :2],
            'loss_l1': loss_l1,
            # 'loss_cons': loss_cons,
            }
