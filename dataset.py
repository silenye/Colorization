import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class VimeoDataset(Dataset):
#     def __init__(self, dataset_name, batch_size=32):
#         self.batch_size = batch_size
#         self.dataset_name = dataset_name
#         self.h = 256
#         self.w = 448
#         self.data_root = 'vimeo_triplet'
#         self.image_root = os.path.join(self.data_root, 'sequences')
#         train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
#         test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
#         with open(train_fn, 'r') as f:
#             self.trainlist = f.read().splitlines()
#         with open(test_fn, 'r') as f:
#             self.testlist = f.read().splitlines()
#         self.load_data()
#
#     def __len__(self):
#         return len(self.meta_data)
#
#     def load_data(self):
#         cnt = int(len(self.trainlist) * 0.95)
#         if self.dataset_name == 'train':
#             self.meta_data = self.trainlist[:cnt]
#         elif self.dataset_name == 'test':
#             self.meta_data = self.testlist
#         else:
#             self.meta_data = self.trainlist[cnt:]
#
#     def aug(self, reference, gt, Ir_1, h, w):
#         ih, iw, _ = reference.shape
#         x = np.random.randint(0, ih - h + 1)
#         y = np.random.randint(0, iw - w + 1)
#         reference = reference[x:x + h, y:y + w, :]
#         Ir_1 = Ir_1[x:x + h, y:y + w, :]
#         gt = gt[x:x + h, y:y + w, :]
#         return reference, gt, Ir_1
#
#     def getimg(self, index):
#         imgpath = os.path.join(self.image_root, self.meta_data[index])
#         imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
#
#         # Load images
#         reference = cv2.imread(imgpaths[0])
#         gt = cv2.imread(imgpaths[1])
#         Ir_1 = cv2.imread(imgpaths[2])
#         return reference, gt, Ir_1
#
#     def __getitem__(self, index):
#         reference, gt, Ir_1 = self.getimg(index)
#         if self.dataset_name == 'train':
#             reference, gt, Ir_1 = self.aug(reference, gt, Ir_1, 224, 224)
#             if random.uniform(0, 1) < 0.5:
#                 reference = reference[:, :, ::-1]
#                 Ir_1 = Ir_1[:, :, ::-1]
#                 gt = gt[:, :, ::-1]
#             if random.uniform(0, 1) < 0.5:
#                 reference = reference[::-1]
#                 Ir_1 = Ir_1[::-1]
#                 gt = gt[::-1]
#             if random.uniform(0, 1) < 0.5:
#                 reference = reference[:, ::-1]
#                 Ir_1 = Ir_1[:, ::-1]
#                 gt = gt[:, ::-1]
#             if random.uniform(0, 1) < 0.5:
#                 tmp = Ir_1
#                 Ir_1 = reference
#                 reference = tmp
#         reference = torch.from_numpy(reference.copy()).permute(2, 0, 1)
#         Ir_1 = torch.from_numpy(Ir_1.copy()).permute(2, 0, 1)
#         gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
#         return torch.cat((reference, Ir_1, gt), 0)
class VimeoDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data_root = r'/home/yun/atd_5/train_10k'
        self.train = []
        self.gt = []
        for folder in os.listdir(self.data_root):
            img1 = 'frame1.jpg'
            img2 = 'frame2.jpg'
            img2_edge = 'frame2_edge.jpg'
            img3 = 'frame3.jpg'
            img3_edge = 'frame3_edge.jpg'
            tmp = [os.path.join(self.data_root, folder, img1),
                   os.path.join(self.data_root, folder, img2_edge)]
            self.train.append(tmp)
            self.gt.append(os.path.join(self.data_root, folder, img2))

            tmp = [os.path.join(self.data_root, folder, img2),
                   os.path.join(self.data_root, folder, img3_edge)]
            self.train.append(tmp)
            self.gt.append(os.path.join(self.data_root, folder, img3))


    def __len__(self):
        return len(self.gt)


    def getimg(self, index):
        # Load images
        # print(self.train[index])
        img1 = cv2.imread(self.train[index][0])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (512, 288), interpolation=cv2.INTER_AREA)
        img2 = cv2.imread(self.train[index][1])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, (512, 288), interpolation=cv2.INTER_AREA)
        gt = cv2.imread(self.gt[index])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (512, 288), interpolation=cv2.INTER_AREA)
        if self.dataset_name == 'train':
            if random.uniform(0, 1) < 0.5:
                img1 = img1[:, :, ::-1]
                img2 = img2[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1 = img1[::-1]
                img2 = img2[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                gt = gt[:, ::-1]

        return img1, img2, gt

    def __getitem__(self, index):
        img1, img2, gt = self.getimg(index)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        img2 = torch.from_numpy(img2.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img1, img2, gt), 0)


class VimeoDataset_valid(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data_root = r'/home/yun/atd_5/test_2k_original'
        self.train = []
        self.gt = []
        for folder in os.listdir(self.data_root):
            img1 = 'frame1.jpg'
            img1_2 = 'frame1_2.jpg'
            img2 = 'frame2.jpg'
            img2_3 = 'frame2_3.jpg'
            img3 = 'frame3.jpg'
            img1_edge = 'frame1_edge.jpg'
            img1_2_edge = 'frame1_2_edge.jpg'
            img2_edge = 'frame2_edge.jpg'
            img2_3_edge = 'frame2_3_edge.jpg'
            img3_edge = 'frame3_edge.jpg'
            tmp = [os.path.join(self.data_root, folder, img1),
                   os.path.join(self.data_root, folder, img1_2),
                   os.path.join(self.data_root, folder, img2),
                   os.path.join(self.data_root, folder, img2_3),
                   os.path.join(self.data_root, folder, img3),
                   os.path.join(self.data_root, folder, img1_edge),
                   os.path.join(self.data_root, folder, img1_2_edge),
                   os.path.join(self.data_root, folder, img2_edge),
                   os.path.join(self.data_root, folder, img2_3_edge),
                   os.path.join(self.data_root, folder, img3_edge)]
            self.train.append(tmp)
            self.gt.append(os.path.join(self.data_root, folder, img1_2))
            # tmp = [os.path.join(self.data_root, folder, Ir_1),
            #        os.path.join(self.data_root, folder, img2_edge)]
            # self.train.append(tmp)
            # self.gt.append(os.path.join(self.data_root, folder, Ir_2))
            #
            # tmp = [os.path.join(self.data_root, folder, Ir_2),
            #        os.path.join(self.data_root, folder, img3_edge)]
            # self.train.append(tmp)
            # self.gt.append(os.path.join(self.data_root, folder, img3))

    def __len__(self):
        return len(self.gt)

    def getimg(self, index):
        # Load images
        # print(self.train[index])
        img1 = cv2.imread(self.train[index][0])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (512, 288), interpolation=cv2.INTER_AREA)
        img1_2 = cv2.imread(self.train[index][1])
        img1_2 = cv2.cvtColor(img1_2, cv2.COLOR_BGR2RGB)
        img1_2 = cv2.resize(img1_2, (512, 288), interpolation=cv2.INTER_AREA)
        img2 = cv2.imread(self.train[index][2])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, (512, 288), interpolation=cv2.INTER_AREA)
        img2_3 = cv2.imread(self.train[index][3])
        img2_3 = cv2.cvtColor(img2_3, cv2.COLOR_BGR2RGB)
        img2_3 = cv2.resize(img2_3, (512, 288), interpolation=cv2.INTER_AREA)
        img3 = cv2.imread(self.train[index][4])
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img3 = cv2.resize(img3, (512, 288), interpolation=cv2.INTER_AREA)

        img1_edge = cv2.imread(self.train[index][5])
        img1_edge = cv2.cvtColor(img1_edge, cv2.COLOR_BGR2RGB)
        img1_edge = cv2.resize(img1_edge, (512, 288), interpolation=cv2.INTER_AREA)
        img1_2_edge = cv2.imread(self.train[index][6])
        img1_2_edge = cv2.cvtColor(img1_2_edge, cv2.COLOR_BGR2RGB)
        img1_2_edge = cv2.resize(img1_2_edge, (512, 288), interpolation=cv2.INTER_AREA)
        img2_edge = cv2.imread(self.train[index][7])
        img2_edge = cv2.cvtColor(img2_edge, cv2.COLOR_BGR2RGB)
        img2_edge = cv2.resize(img2_edge, (512, 288), interpolation=cv2.INTER_AREA)
        img2_3_edge = cv2.imread(self.train[index][8])
        img2_3_edge = cv2.cvtColor(img2_3_edge, cv2.COLOR_BGR2RGB)
        img2_3_edge = cv2.resize(img2_3_edge, (512, 288), interpolation=cv2.INTER_AREA)
        img3_edge = cv2.imread(self.train[index][9])
        img3_edge = cv2.cvtColor(img3_edge, cv2.COLOR_BGR2RGB)
        img3_edge = cv2.resize(img3_edge, (512, 288), interpolation=cv2.INTER_AREA)

        gt = img2
        return img1, img1_2_edge, img2_edge, img2_3_edge, img3_edge, gt
        # img = cv2.imread(self.train[index][0])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (512, 288), interpolation=cv2.INTER_AREA)
        #
        # Is = cv2.imread(self.train[index][1])
        # Is = cv2.cvtColor(Is, cv2.COLOR_BGR2RGB)
        # Is = cv2.resize(Is, (512, 288), interpolation=cv2.INTER_AREA)
        #
        # gt = cv2.imread(self.gt[index])
        # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        # gt = cv2.resize(gt, (512, 288), interpolation=cv2.INTER_AREA)
        # return img, Is, gt


    def __getitem__(self, index):
        img1, img1_2_edge, img2_edge, img2_3_edge, img3_edge, gt = self.getimg(index)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        img1_2_edge = torch.from_numpy(img1_2_edge.copy()).permute(2, 0, 1)
        img2_edge = torch.from_numpy(img2_edge.copy()).permute(2, 0, 1)
        img2_3_edge = torch.from_numpy(img2_3_edge.copy()).permute(2, 0, 1)
        img3_edge = torch.from_numpy(img3_edge.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img1, img1_2_edge, img2_edge, img2_3_edge, img3_edge, gt), 0)
        # img, Is, gt = self.getimg(index)
        # img = torch.from_numpy(img.copy()).permute(2, 0, 1)
        # Is = torch.from_numpy(Is.copy()).permute(2, 0, 1)
        # gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        # return torch.cat((img, Is, gt), 0)

class VimeoDataset_one_to_three(Dataset):
    # (frame1, frame2_edge) -> color_frame2
    # (color_frame2, frame3_edge) -> color frame3
    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        self.data_root = r'/home/yun/atd_5/test_2k_original'
        self.train = []
        self.gt = []
        for folder in os.listdir(self.data_root):
            img1 = 'frame1.jpg'
            img1_edge = 'frame1_edge.jpg'
            img1_2 = 'frame1_2.jpg'
            img1_2_edge = 'frame1_2_edge.jpg'
            img2 = 'frame2.jpg'
            img2_edge = 'frame2_edge.jpg'
            img2_3 = 'frame2_3.jpg'
            img2_3_edge = 'frame2_3_edge.jpg'
            img3 = 'frame3.jpg'
            img3_edge = 'frame3_edge.jpg'
            tmp = [os.path.join(self.data_root, folder, img1),
                   os.path.join(self.data_root, folder, img1_edge),
                   os.path.join(self.data_root, folder, img1_2),
                   os.path.join(self.data_root, folder, img1_2_edge),
                   os.path.join(self.data_root, folder, img2),
                   os.path.join(self.data_root, folder, img2_edge),
                   os.path.join(self.data_root, folder, img2_3),
                   os.path.join(self.data_root, folder, img2_3_edge),
                   os.path.join(self.data_root, folder, img3),
                   os.path.join(self.data_root, folder, img3_edge)]
            self.train.append(tmp)
            self.gt.append(os.path.join(self.data_root, folder, img2))


    def __len__(self):
        return len(self.gt)

    def getimg(self, index):
        # Load images
        # print(self.train[index])
        img1 = cv2.imread(self.train[index][0])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (512, 288), interpolation=cv2.INTER_AREA)
        img1_edge = cv2.imread(self.train[index][1])
        img1_edge = cv2.cvtColor(img1_edge, cv2.COLOR_BGR2RGB)
        img1_edge = cv2.resize(img1_edge, (512, 288), interpolation=cv2.INTER_AREA)
        img1_2 = cv2.imread(self.train[index][2])
        img1_2 = cv2.cvtColor(img1_2, cv2.COLOR_BGR2RGB)
        img1_2 = cv2.resize(img1_2, (512, 288), interpolation=cv2.INTER_AREA)
        img1_2_edge = cv2.imread(self.train[index][3])
        img1_2_edge = cv2.cvtColor(img1_2_edge, cv2.COLOR_BGR2RGB)
        img1_2_edge = cv2.resize(img1_2_edge, (512, 288), interpolation=cv2.INTER_AREA)
        img2 = cv2.imread(self.train[index][4])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, (512, 288), interpolation=cv2.INTER_AREA)
        img2_edge = cv2.imread(self.train[index][5])
        img2_edge = cv2.cvtColor(img2_edge, cv2.COLOR_BGR2RGB)
        img2_edge = cv2.resize(img2_edge, (512, 288), interpolation=cv2.INTER_AREA)
        img2_3 = cv2.imread(self.train[index][6])
        img2_3 = cv2.cvtColor(img2_3, cv2.COLOR_BGR2RGB)
        img2_3 = cv2.resize(img2_3, (512, 288), interpolation=cv2.INTER_AREA)
        img2_3_edge = cv2.imread(self.train[index][7])
        img2_3_edge = cv2.cvtColor(img2_3_edge, cv2.COLOR_BGR2RGB)
        img2_3_edge = cv2.resize(img2_3_edge, (512, 288), interpolation=cv2.INTER_AREA)
        img3 = cv2.imread(self.train[index][8])
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img3 = cv2.resize(img3, (512, 288), interpolation=cv2.INTER_AREA)
        img3_edge = cv2.imread(self.train[index][9])
        img3_edge = cv2.cvtColor(img3_edge, cv2.COLOR_BGR2RGB)
        img3_edge = cv2.resize(img3_edge, (512, 288), interpolation=cv2.INTER_AREA)
        gt = cv2.imread(self.gt[index])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (512, 288), interpolation=cv2.INTER_AREA)

        return img1, img1_2_edge, img2_edge, img2_3_edge, img3_edge, gt
        # return Ir_1, img1_2, Ir_2, img2_3, img3, gt

    def __getitem__(self, index):
        img1, img1_2_edge, img2_edge, img2_3_edge, img3_edge, gt = self.getimg(index)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        img1_2_edge = torch.from_numpy(img1_2_edge.copy()).permute(2, 0, 1)
        img2_edge = torch.from_numpy(img2_edge.copy()).permute(2, 0, 1)
        img2_3_edge = torch.from_numpy(img2_3_edge.copy()).permute(2, 0, 1)
        img3_edge = torch.from_numpy(img3_edge.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img1, img1_2_edge, img2_edge, img2_3_edge, img3_edge, gt), 0)

# if __name__ == '__main__':
#     dataset = VimeoDataset('train')
#     train_data = DataLoader(dataset, batch_size=2, num_workers=0, pin_memory=True, drop_last=True)
#     i, data = next(enumerate(train_data))
#     pass