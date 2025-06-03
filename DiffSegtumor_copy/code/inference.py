import os
import argparse
from tqdm import tqdm
import numpy as np
import math
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='brats_2d')
parser.add_argument('--exp', type=str, default='Exp_tumorseg_brats/diffusion/fold1')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('-g', '--gpu', type=str,  default='1')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F

from DiffVNet.diff_vnet_2d import DiffVNet
from utils import  maybe_mkdir
from utils import config
from data.data_loaders import MergedNiiDataset
from torch.utils.data import DataLoader
from utils import  maybe_mkdir, fetch_data
from medpy import metric

import cv2
 
config = config.Config(args.task)

if __name__ == '__main__':
    stride_dict = {
        0: (16, 16),
        1: (32, 32),
        2: (64, 64),
    }
    stride = stride_dict[args.speed]
    snapshot_path = f'./logs/{args.exp}/'
    test_save_path = f'./logs/{args.exp}/predictions/'
    txt_path = "./logs/"+args.exp+"/evaluation_res.txt"
    print("\n Evaluating...")
    fw = open(txt_path, 'w')

    maybe_mkdir(test_save_path)
    print(snapshot_path)

    model = DiffVNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=False
    ).cuda()


    ckpt_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')

    with torch.no_grad():
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        print(f'load checkpoint from {ckpt_path}')

        test_set = MergedNiiDataset(
            task = config.task,
            split="test",
            selected_modalities=['t2','seg'],
            transform=None,
            is_val=True,
            num_cls = config.num_cls,
            suffix ='npy'

        )

        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            drop_last=False
        )
        
        values = np.zeros((len(test_loader), config.num_cls, 2)) # dice and asd
   
        for step, batch in enumerate(tqdm(test_loader)):
            image, label = fetch_data(batch)
            # p_u_theta = model(image, pred_type="D_psi_l")
            image = image[0,0,:,:].cpu().numpy()
            label = label[0,0,:,:].cpu().numpy()

            h, w = image.shape
            ph, pw = config.patch_size
            sh, sw = stride



            pad_h = max((ph - h), 0)
            pad_w = max((pw - w), 0)
            padding_flag = pad_h > 0 or pad_w > 0
            # print('padding_flag',padding_flag,'h, w ',h, w )
            if padding_flag:
                image = np.pad(image, [(pad_h, pad_h), (pad_w, pad_w)], mode='constant', constant_values=0)

            # print("image",image.shape,'ph, pw ',ph, pw )
            image = image[np.newaxis]  # shape: (1, H, W)
            _, H, W = image.shape
            
            if H == ph and W == pw:
                test_patch = torch.from_numpy(image).unsqueeze(0).cuda().float()  # shape: (1, 1, ph, pw)
                y1 = model(test_patch, pred_type="D_psi_l")
                y = F.softmax(y1, dim=1).cpu().numpy()[0]  # shape: (num_cls, ph, pw)
                score_map = y
            else:
                
                sx = math.ceil((H - ph) / sh) + 1
                sy = math.ceil((W - pw) / sw) + 1
                score_map = np.zeros((config.num_cls, H, W), dtype=np.float32)
                cnt = np.zeros((H, W), dtype=np.float32)
                for x in range(sx):
                    xs = min(sh * x, H - ph)
                    for y in range(sy):
                        ys = min(sw * y, W - pw)
                        patch = image[:, xs:xs + ph, ys:ys + pw]
                        patch = torch.from_numpy(patch).unsqueeze(0).cuda().float()  # shape: (1, 1, ph, pw)
                        y1 = model(patch, pred_type="D_psi_l")
                        y = F.softmax(y1, dim=1).cpu().numpy()[0]  # shape: (num_cls, ph, pw)
                        score_map[:, xs:xs + ph, ys:ys + pw] += y
                        cnt[xs:xs + ph, ys:ys + pw] += 1

                # Normalize overlapping regions
                score_map = score_map / np.maximum(cnt[None, ...], 1e-5)

            pred = np.argmax(score_map, axis=0)
            cv2.imwrite(f'{test_save_path}/{step}.png', pred/3*255)
            cv2.imwrite(f'{test_save_path}/{step}_label.png', label/3*255)

            for i in range(config.num_cls):
                pred_i = (pred == i)
                label_i = (label == i)
                if pred_i.sum() > 0 and label_i.sum() > 0:
                    dice = metric.binary.dc(pred == i, label == i) * 100
                    hd95 = metric.binary.asd(pred == i, label == i)
                    values[step][i-1] = np.array([dice, hd95])
                elif pred_i.sum() > 0 and label_i.sum() == 0:
                    dice, hd95 = 0, 128
                elif pred_i.sum() == 0 and label_i.sum() > 0:
                    dice, hd95 =  0, 128
                elif pred_i.sum() == 0 and label_i.sum() == 0:
                    dice, hd95 =  1, 0

                    values[step][i-1] = np.array([dice, hd95])

        values_mean_cases = np.mean(values, axis=0)
        fw.write("------ Dice ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,0],1)) + '\n')
        fw.write("------ ASD ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,1],1)) + '\n')
        fw.write('Average Dice:'+str(np.mean(values_mean_cases, axis=0)[0]) + '\n')
        fw.write('Average  ASD:'+str(np.mean(values_mean_cases, axis=0)[1]) + '\n')
        fw.write("=================================")
        print("------ Dice ------")
        print(np.round(values_mean_cases[:,0],1))
        print("------ ASD ------")
        print(np.round(values_mean_cases[:,1],1))
        print(np.mean(values_mean_cases, axis=0)[0], np.mean(values_mean_cases, axis=0)[1])

