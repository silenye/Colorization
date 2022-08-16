import torch
import torch.nn as nn
import torch.nn.functional as F
from models.warplayer import warp
# from get_flow import flow2rgb
import matplotlib.pyplot as plt
import numpy as np
from models.unet_model import UNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
   )

class DegCNN(nn.Module):
    def __init__(self):
        super(DegCNN, self).__init__()
        self.conv0 = conv(3, 32, 3, 2, 1)
        self.conv1 = conv(32, 32, 3, 2, 1)
        self.conv2 = conv(32, 32, 3, 2, 1)
        self.conv3 = conv(32, 32, 3, 2, 1)
        self.deconv = nn.Sequential(
            nn.Dropout2d(0.95),
            nn.ConvTranspose2d(4 * 32, 32, 4, 2, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )
                                             
    def forward(self, x):
        f0 = self.conv0(x)
        f1 = self.conv1(f0)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f1 = F.interpolate(f1, scale_factor=2.0, mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, scale_factor=4.0, mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, scale_factor=8.0, mode="bilinear", align_corners=False)
        return self.deconv(torch.cat((f0, f1, f2, f3), 1))

class Flow_Estimate_Block(nn.Module):
    def __init__(self, in_planes, c=64):
        super(Flow_Estimate_Block, self).__init__()
        self.conv0 = nn.Sequential(
            conv_bn(in_planes, c//2, 3, 2, 1),
            conv_bn(c//2, c, 3, 2, 1),
            conv_bn(c, 2*c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv_bn(2*c, 2*c),
            conv_bn(2*c, 2*c),
            conv_bn(2*c, 2*c),
            conv_bn(2*c, 2*c),
            conv_bn(2*c, 2*c),
            conv_bn(2*c, 2*c),
        )
        self.lastconv = nn.ConvTranspose2d(2*c, 4, 4, 2, 1)

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)        
        if flow is not None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            if flow.shape != x.shape:
                flow = flow[:, :, :x.shape[2], :x.shape[3]]
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat) + feat
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale*4, mode="bilinear", align_corners=False)
        flow = tmp[:, :2] * scale * 4
        mask = tmp[:, 2:3]
        return flow, mask
        
class Flow_Estimate_Net(nn.Module):
    def __init__(self):
        super(Flow_Estimate_Net, self).__init__()
        self.block0 = Flow_Estimate_Block(6, c=128)
        self.block1 = Flow_Estimate_Block(12, c=128)
        self.block2 = Flow_Estimate_Block(12, c=128)
        self.degrad = DegCNN()
        # Contextual Refinement context + decode
        self.context0 = nn.Sequential(
            conv(3, 16, 3, 2, 1),
            conv(16, 32, 3, 2, 1),
        )
        self.context1 = nn.Sequential(
            conv(3, 16, 3, 2, 1),
            conv(16, 32, 3, 2, 1),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )
        self.unet = UNet(6, 3)

    def calflow(self, reference, sketch, scale):
        flow = None
        block_list = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = block_list[i](torch.cat((reference, sketch, warped_img, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = block_list[i](torch.cat((reference, sketch), 1), None, scale=scale[i])
            warped_img = warp(reference, flow)
        flow_down = F.interpolate(flow, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        c0 = warp(self.context0(reference), flow_down)   # 1/4尺度下warp
        c1 = self.context1(warped_img)
        warped_img = warped_img + self.decode(torch.cat((c0, c1), 1))
        return flow, mask, torch.clamp(warped_img, 0, 1)
    
    def forward(self, x, deg=None, gt=None, scale=[4, 2, 1], training=False, blend=False):
        # if training:
        #     deg = self.degrad(gt)
        #     loss_cons = (gt - deg).abs().mean()
        # else:
        #     loss_cons = torch.tensor([0])
        img_list = []
        N = x.shape[1] // 3 # 代表有几张参考图片
        for i in range(N):
            img_list.append(x[:, i*3:i*3+3])
        warped_list = []
        merged = []
        mask_list = []
        flow_list = []
        flow = None
        for i in range(N):
            f, m, img = self.calflow(img_list[i], deg.detach(), scale)
            mask_list.append(m)
            warped_list.append(img)
            flow_list.append(f)
        if blend:
            N += 1
            mask_list.append(m * 0)
            warped_list.append(deg)
        mask = F.softmax(torch.clamp(torch.cat(mask_list, 1), -4, 4), dim=1)
        merged = 0
        for i in range(N):
            merged += warped_list[i] * mask[:, i:i+1]
        u_merged = self.unet(torch.cat((merged, deg.detach()), 1))
        return merged, u_merged
        # return flow_list, mask_list
