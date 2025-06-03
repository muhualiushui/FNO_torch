import os
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.config import Config
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




class MergedNiiDataset(Dataset):
    def __init__(self, task,split='train', selected_modalities=None, transform=None, is_val=False,num_cls = 4,suffix='npy'):
        """
        Dataset for loading merged npy files per modality.

        """
        self.split = split
        self.cfg = Config(task)
        self.modalities = selected_modalities if selected_modalities else self.cfg.modalities
        self.transform = transform
        self.is_val = is_val
        self.num_cls = num_cls  
        self.suffix =suffix
        self.data = {}
        self.length = None

        # === Load .npy files using memory mapping ===
        if suffix =="npy":
            for mod in self.modalities:
                path = os.path.join(self.cfg.save_dir, f"{split}_merged_{mod}.npy")
                self.data[mod] = np.load(path, mmap_mode='r')
        elif suffix =="gz":
            for mod in self.modalities:
                path = os.path.join(self.cfg.save_dir, f"{split}_merged_{mod}.nii.gz")
                self.data[mod] = nib.load(path).get_fdata()            
        self.total_slices = self.data[self.modalities[0]].shape[2]

    def __len__(self):
        return self.total_slices

    def __getitem__(self, index):
        """
        Extract 2D slice from each modality at given Z index.
        """
        # Input: Stack selected modalities as channels
        input_modalities = [mod for mod in self.modalities if mod != 'seg']
        # image = np.stack([self.data[mod][:, :, index] for mod in input_modalities], axis=0)  # [C, H, W]
        image = self.data[input_modalities[0]][:, :, index]
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)

        if 'seg' in self.modalities:
            label = self.data['seg'][:, :, index].astype(np.int8)
            label[label == 4] = 3
        else:
            label[:] = 0
        image = image.astype(np.float32)
        label = label.astype(np.int8)
        sample = {'image': image, 'label': label}

        if not self.is_val:
            if self.transform:
                sample = self.transform(sample)
        else:
            sample['image'] = torch.from_numpy(np.expand_dims(sample['image'], 0)).float()
            sample['label'] = torch.from_numpy(sample['label'])

        # # ============== training ==============
        # if not self.is_val:
        #     if self.transform:
        #         sample = self.transform(sample)
        #     if 'label' in sample:
        #         sample['label'] = F.one_hot(sample['label'].long(), self.num_cls).permute(2, 0, 1).float()
        # # ============== validation ==============
    
        return sample             


    



        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 1, 1)
        # plt.imshow(image[0], cmap='gray')
        # plt.title(f'NIfTI [{self.suffix}]')
        # plt.axis('off')
        # plt.savefig(self.suffix+'.png', dpi=150, bbox_inches='tight')
        # plt.close()

