import numpy as np
import os
import torch.nn.functional as F
os.environ["RANK"] = "0"
os.environ['WORLD_SIZE'] = '1'
import argparse
import matplotlib.pyplot as plt
from models.model import Model
from dataset import *
def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

device = torch.device("cuda")

log_path = 'train_log'

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

    inference_img = r'/media/yun/4t/work/yyf/flow_estimate_net/Colorization/visio/frame2.jpg'
    edge_img = r'/media/yun/4t/work/yyf/flow_estimate_net/Colorization/visio/frame3_edge.jpg'
    gt_img = r'/media/yun/4t/work/yyf/flow_estimate_net/Colorization/visio/frame3.jpg'
    img1 = cv2.imread(inference_img)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (512, 288), interpolation=cv2.INTER_AREA)
    img2 = cv2.imread(edge_img)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (512, 288), interpolation=cv2.INTER_AREA)
    gt = cv2.imread(gt_img)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = cv2.resize(gt, (512, 288), interpolation=cv2.INTER_AREA)

    img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1) / 255
    img2 = torch.from_numpy(img2.copy()).permute(2, 0, 1) / 255
    gt = torch.from_numpy(gt.copy()).permute(2, 0, 1) / 255

    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    flow_list, mask_list = model.get_flow(img1, img2)
    flow_offset = flow_list[0][0].permute(1, 2, 0).detach().cpu().numpy()   # [2, 288, 512]
    rgb_flow = flow2rgb(flow_offset)
    mask = F.softmax(torch.clamp(torch.cat(mask_list, 1), -4, 4), dim=1)
    mask = mask[0].permute(1, 2, 0).detach().cpu().numpy()
    mask *= 255
    mask = np.concatenate((mask, mask, mask), axis=2)
    plt.imshow(rgb_flow)
    # plt.show()
    # plt.imsave('./2_3_block3_output_flow.jpg', rgb_flow)
    # pass