from re import sub
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
import h5py
import cv2
import torchvision.transforms as T

class NYUDataset(Dataset):
    def __init__(self, root_dir, split='train', shuffle=False, img_num=1, visible_img=1, focus_dist=[0.1,.15,.3,0.7,1.5], recon_all=True, 
                    RGBFD=False, DPT=False, AIF=False, scale=2, norm=False, near=0.1, far=1., trans=False, resize=256):
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.img_num = img_num
        self.visible_img = visible_img
        self.focus_dist = torch.Tensor(focus_dist)
        self.recon_all = recon_all
        self.RGBFD = RGBFD
        self.DPT = DPT
        self.AIF = AIF
        # self.scale = scale
        self.norm = norm
        self.trans = trans
        self.near = near
        self.far = far
        if resize is not None:
            self.transform = T.Resize(resize/scale)
        else:
            self.transform = None

        self.aif_path = os.path.join(self.root_dir, f'{split}_rgb')
        self.dpt_path = os.path.join(self.root_dir, f'{split}_depth')
        if self.norm:
            self.all_path = os.path.join(self.root_dir, f'{split}_fs5')
        elif self.trans:
            self.all_path = os.path.join(self.root_dir, f'{split}_fs5_orig_trans')
        else:
            self.all_path = os.path.join(self.root_dir, f'{split}_fs5_orig')
            # self.all_path = os.path.join(self.root_dir, f'{split}_fs5_cam_trans')
        
        ##### Load and sort all images
        self.imglist_all = [f for f in os.listdir(self.all_path) if os.path.isfile(os.path.join(self.all_path, f))]
        self.imglist_dpt = [f for f in os.listdir(self.dpt_path) if os.path.isfile(os.path.join(self.dpt_path, f))]
        self.imglist_aif = [f for f in os.listdir(self.aif_path) if os.path.isfile(os.path.join(self.aif_path, f))]

        self.n_stack = len(self.imglist_aif)
        if split == 'train':
            print(f"{self.visible_img} out of {self.img_num} images per sample are visible for input")
        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.imglist_aif.sort()

    def __len__(self):
        return self.n_stack

    def __getitem__(self, idx):
        img_idx = idx * self.img_num

        sub_idx = np.arange(self.img_num)
        if self.shuffle:
            np.random.shuffle(sub_idx)
        input_idx = sub_idx[:self.visible_img]
        if self.recon_all:
            output_idx = sub_idx
        else:
            output_idx = sub_idx[self.visible_img:]

        mats_input = []
        mats_output = []

        for i in sub_idx:
            img_all = cv2.imread(os.path.join(self.all_path, self.imglist_all[img_idx + i])) / 255.
            mat_all = img_all.copy().astype(np.float32)
            if self.transform is not None:
                mat_all = self.transform(mat_all)

            if i in output_idx:    
                mats_output.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))
            if self.RGBFD and i in input_idx:   
                mat_fd = self.focus_dist[i].view(-1, 1, 1).expand(*mat_all.shape[:-1], 1).numpy()
                mat_all = np.concatenate([mat_all, mat_fd], axis=-1)                 
                mats_input.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))

        data = dict(output=torch.cat(mats_output), output_fd=self.focus_dist[output_idx])

        if self.RGBFD:
            data.update(rgb_fd = torch.cat(mats_input))

        if self.DPT:
            img_dpt = Image.open(os.path.join(self.dpt_path, self.imglist_dpt[idx]))
            img_dpt = np.asarray(img_dpt, dtype=np.float32)
            if self.transform is not None:
                img_dpt = self.transform(img_dpt)
            if self.norm:
                img_dpt = (img_dpt - img_dpt.min())/(img_dpt.max() - img_dpt.min()) 
                img_dpt = img_dpt * (self.far - self.near) + self.near
            else:
                img_dpt = np.clip(img_dpt / 1e4, self.near, self.far)
            H, W = img_dpt.shape
            # img_dpt = cv2.resize(img_dpt, (W//self.scale, H//self.scale))
            mat_dpt = img_dpt.copy()
            mat_dpt = torch.from_numpy(mat_dpt).unsqueeze(0)
            data.update(dpt = mat_dpt)

        if self.AIF:
            im = cv2.imread(os.path.join(self.aif_path, self.imglist_aif[idx]))
            H, W, C = im.shape
            # im = cv2.resize(im, (W//self.scale, H//self.scale))/255.
            if self.transform is not None:
                im = self.transform(im)
            img_aif = np.array(im)
            mat_aif = img_aif.copy().astype(np.float32)
            data.update(aif = torch.from_numpy(mat_aif.transpose(2, 0, 1)))

        return data

class NYUFS100Dataset(Dataset):
    def __init__(self, root_dir, focus_dist, split='train', shuffle=True, img_num=100, visible_img=5, recon_all=True, 
                    RGBFD=False, DPT=False, AIF=False, scale=2, near=0.1, far=10.):
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.img_num = img_num
        self.visible_img = visible_img
        self.focus_dist = torch.Tensor(focus_dist)
        self.recon_all = recon_all
        self.RGBFD = RGBFD
        self.DPT = DPT
        self.AIF = AIF
        self.scale = scale
        self.near = near
        self.far = far

        self.aif_path = os.path.join(self.root_dir, f'{split}_rgb')
        self.dpt_path = os.path.join(self.root_dir, f'{split}_depth')
        # print('Modified DATASET!!!!!')
        # self.all_path = os.path.join(self.root_dir, f'{split}_fs100_cam_trans')
        self.all_path = os.path.join(self.root_dir, f'{split}_fs100_orig')
        
        ##### Load and sort all images
        self.imglist_all = [f for f in os.listdir(self.all_path) if os.path.isfile(os.path.join(self.all_path, f))]
        self.imglist_dpt = [f for f in os.listdir(self.dpt_path) if os.path.isfile(os.path.join(self.dpt_path, f))]
        self.imglist_aif = [f for f in os.listdir(self.aif_path) if os.path.isfile(os.path.join(self.aif_path, f))]

        self.n_stack = len(self.imglist_aif)
        if split == 'train':
            print(f"{self.visible_img} out of {self.img_num} images per sample are visible for input")
        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.imglist_aif.sort()

    def __len__(self):
        return self.n_stack

    def __getitem__(self, idx):
        img_idx = idx * self.img_num

        sub_idx = np.arange(self.img_num)
        n_bin = self.img_num // self.visible_img
        anchor = np.arange(0, self.img_num, n_bin)[:self.visible_img]
        if self.shuffle:
            input_idx = [sub_idx[i + np.random.randint(0, n_bin)] for i in anchor]
        else:
            input_idx = [sub_idx[i + n_bin // 2] for i in anchor]
        if self.recon_all:
            output_idx = sub_idx
        else:
            output_idx = input_idx

        mats_input = []
        mats_output = []

        for i in sub_idx:
            img_all = cv2.imread(os.path.join(self.all_path, self.imglist_all[img_idx + i])) / 255.
            mat_all = img_all.copy().astype(np.float32)  
            if i in output_idx:
                mats_output.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))
            if self.RGBFD and i in input_idx:   
                mat_fd = self.focus_dist[i].view(-1, 1, 1).expand(*mat_all.shape[:-1], 1).numpy()
                mat_all = np.concatenate([mat_all, mat_fd], axis=-1)                 
                mats_input.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))

        data = dict(output=torch.cat(mats_output), output_fd=self.focus_dist[output_idx])

        if self.RGBFD:
            data.update(rgb_fd = torch.cat(mats_input))

        if self.DPT:
            img_dpt = Image.open(os.path.join(self.dpt_path, self.imglist_dpt[idx]))
            img_dpt = np.asarray(img_dpt, dtype=np.float32)
            img_dpt = np.clip(img_dpt / 1e4, self.near, self.far)
            H, W = img_dpt.shape
            img_dpt = cv2.resize(img_dpt, (W//self.scale, H//self.scale))
            mat_dpt = img_dpt.copy()
            mat_dpt = torch.from_numpy(mat_dpt).unsqueeze(0)
            data.update(dpt = mat_dpt)

        if self.AIF:
            im = cv2.imread(os.path.join(self.aif_path, self.imglist_aif[idx]))
            H, W, C = im.shape
            im = cv2.resize(im, (W//self.scale, H//self.scale))/255.
            img_aif = np.array(im)
            mat_aif = img_aif.copy().astype(np.float32)
            data.update(aif = torch.from_numpy(mat_aif.transpose(2, 0, 1)))

        return data

class DSLRDataset(Dataset):
    def __init__(self, root_dir, split='train', shuffle=False, img_num=1, visible_img=1, focus_dist=[1, 1.5, 2.5, 4, 6], recon_all=True, 
                    RGBFD=False, DPT=False, AIF=False, scale=1, near=0.1, far=10.):
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.img_num = img_num
        self.visible_img = visible_img
        self.focus_dist = torch.Tensor(focus_dist)
        self.recon_all = recon_all
        self.RGBFD = RGBFD
        self.DPT = DPT
        self.AIF = AIF
        self.scale = scale
        self.near = near
        self.far = far

        self.aif_path = os.path.join(self.root_dir, 'rgb', split)
        self.dpt_path = os.path.join(self.root_dir, 'depth', split)
        self.all_path = os.path.join(self.root_dir, 'fs5', split)

        ##### Load and sort all images
        self.imglist_all = [f for f in os.listdir(self.all_path) if os.path.isfile(os.path.join(self.all_path, f))]
        self.imglist_dpt = [f for f in os.listdir(self.dpt_path) if os.path.isfile(os.path.join(self.dpt_path, f))]
        self.imglist_aif = [f for f in os.listdir(self.aif_path) if os.path.isfile(os.path.join(self.aif_path, f))]

        self.n_stack = len(self.imglist_aif)
        if split == 'train':
            print(f"{self.visible_img} out of {self.img_num} images per sample are visible for input")
        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.imglist_aif.sort()

    def __len__(self):
        return self.n_stack

    def __getitem__(self, idx):
        img_idx = idx * self.img_num

        sub_idx = np.arange(self.img_num)
        if self.shuffle:
            np.random.shuffle(sub_idx)
        input_idx = sub_idx[:self.visible_img]
        if self.recon_all:
            output_idx = sub_idx
        else:
            output_idx = sub_idx[self.visible_img:]

        mats_input = []
        mats_output = []

        for i in sub_idx:
            img_all = cv2.imread(os.path.join(self.all_path, self.imglist_all[img_idx + i])) / 255.
            mat_all = img_all[:208, :320].copy().astype(np.float32)
            if i in output_idx:    
                mats_output.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))
            if self.RGBFD and i in input_idx:   
                mat_fd = self.focus_dist[i].view(-1, 1, 1).expand(*mat_all.shape[:-1], 1).numpy()
                mat_all = np.concatenate([mat_all, mat_fd], axis=-1)                 
                mats_input.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))

        data = dict(output=torch.cat(mats_output), output_fd=self.focus_dist[output_idx])

        if self.RGBFD:
            data.update(rgb_fd = torch.cat(mats_input))

        if self.DPT:
            img_dpt = Image.open(os.path.join(self.dpt_path, self.imglist_dpt[idx]))
            img_dpt = np.asarray(img_dpt, dtype=np.float32)[:416, :640]
            img_dpt = np.clip(img_dpt / 1e3, self.near, self.far)
            H, W = img_dpt.shape
            img_dpt = cv2.resize(img_dpt, (W//self.scale, H//self.scale))
            mat_dpt = img_dpt.copy()
            mat_dpt = torch.from_numpy(mat_dpt).unsqueeze(0)
            data.update(dpt = mat_dpt)

        if self.AIF:
            im = cv2.imread(os.path.join(self.aif_path, self.imglist_aif[idx]))
            H, W, C = im.shape
            im = cv2.resize(im, (W//self.scale, H//self.scale))/255.
            img_aif = np.array(im[:416//self.scale, :640//self.scale])
            mat_aif = img_aif.copy().astype(np.float32)
            data.update(aif = torch.from_numpy(mat_aif.transpose(2, 0, 1)))

        return data

class MobileDFD(Dataset):
    def __init__(self, root_dir, recon_all=True, visible_img=5, RGBFD=True, scale=1, near=0.1, far=10.):
        self.root_dir = root_dir
        self.visible_img = visible_img
        self.recon_all = recon_all
        self.RGBFD = RGBFD
        self.scale = scale
        self.near = near
        self.far = far

        ##### Load and sort all images
        self.imglist_dir = os.listdir(self.root_dir)

        self.n_stack = len(self.imglist_dir) 
        print(f"{self.visible_img} images per sample are visible for input")
        self.imglist_dir.sort()

    def __len__(self):
        return self.n_stack

    def __getitem__(self, idx):
        image_file = os.path.join(self.root_dir, self.imglist_dir[idx])
        imglist_all = [f for f in os.listdir(image_file) if f[-4:]=='.jpg' and f[0]=='a']
        focus_dist = np.genfromtxt(os.path.join(image_file, 'focus_dpth.txt')).astype(np.float32) / 39.37
        if self.imglist_dir[idx] == 'metals':  # metals' distance esitimation is oppsoite, as we only care about relative dist, directly take minus
            focus_dist = focus_dist[::-1]
        focus_dist = torch.from_numpy(focus_dist.copy())
        imglist_all.sort()
        
        img_num = len(imglist_all)
        # very_far = sum(focus_dist > 20).numpy()
        # sub_idx = np.linspace(very_far, img_num-1, self.visible_img).round().astype(np.int)
        # sub_idx = [img_num-1, img_num-6, img_num-12, img_num-12, img_num-18]
        sub_idx = np.arange(img_num)
        
        input_idx = sub_idx
        if self.recon_all:
            output_idx = np.arange(img_num)
        else:
            output_idx = sub_idx

        mats_input = []
        mats_output = []

        for i in output_idx:
            mat_all = Image.open(os.path.join(image_file, imglist_all[i]))
            mat_all = np.asarray(mat_all, dtype=np.float32) / 255.
            H, W, C = mat_all.shape
            print(H, W)
            mat_all = cv2.resize(mat_all, (W//self.scale//16*16, H//self.scale//16*16))
            mats_output.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))
            if self.RGBFD and i in input_idx:   
                mat_fd = focus_dist[i].view(-1, 1, 1).expand(*mat_all.shape[:-1], 1).numpy()
                mat_all = np.concatenate([mat_all, mat_fd], axis=-1)
                mats_input.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))

        data = dict(output=torch.cat(mats_output), output_fd=focus_dist[output_idx])

        if self.RGBFD:
            data.update(rgb_fd = torch.cat(mats_input))

        return data

class SelfCollectedDS(Dataset):
    def __init__(self, root_dir, split='train', shuffle=False, img_num=1, visible_img=1, focus_dist=[0.5, 0.8, 1.2, 1.5, 3.0], recon_all=True, 
                    RGBFD=True, scale=10, near=0.1, far=6.):
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.img_num = img_num
        self.visible_img = visible_img
        self.focus_dist = torch.Tensor(focus_dist)
        self.recon_all = recon_all
        self.RGBFD = RGBFD
        self.scale = scale
        self.near = near
        self.far = far

        self.all_path = self.root_dir
        self.H_clip = [0, 33, 48, 77, 80]
        self.V_clip = [0, 50, 72, 116, 120]

        ##### Load and sort all images
        self.imglist_all = [f for f in os.listdir(self.all_path) if os.path.isfile(os.path.join(self.all_path, f)) and f[-4:]=='.tif']

        self.n_stack = len(self.imglist_all) // self.img_num
        if split == 'train':
            print(f"{self.visible_img} out of {self.img_num} images per sample are visible for input")
        self.imglist_all.sort()

    def __len__(self):
        return self.n_stack

    def __getitem__(self, idx):
        img_idx = idx * self.img_num

        sub_idx = np.arange(self.img_num)
        n_bin = self.img_num // self.visible_img
        anchor = np.arange(0, self.img_num, n_bin)[:self.visible_img]
        if self.shuffle:
            input_idx = [sub_idx[i + np.random.randint(0, n_bin)] for i in anchor]
        else:
            input_idx = [sub_idx[i + n_bin // 2] for i in anchor]
        
        if self.recon_all:
            output_idx = sub_idx
        else:
            output_idx = sub_idx[self.visible_img:]

        mats_input = []
        mats_output = []

        for i in sub_idx:
            mat_all = Image.open(os.path.join(self.all_path, self.imglist_all[i]))
            mat_all = np.asarray(mat_all, dtype=np.float32) / 255.
            H, W, C = mat_all.shape
            #if i != 0:
            #    mat_all = mat_all[self.H_clip[i]: -self.H_clip[i], self.V_clip[i]: -self.V_clip[i]]
            mat_all = cv2.resize(mat_all, (W//16*16, H//16*16))
            if i in output_idx:    
                mats_output.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))
            if self.RGBFD and i in input_idx:   
                mat_fd = self.focus_dist[i].view(-1, 1, 1).expand(*mat_all.shape[:-1], 1).numpy()
                mat_all = np.concatenate([mat_all, mat_fd], axis=-1)                 
                mats_input.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))

        data = dict(output=torch.cat(mats_output), output_fd=self.focus_dist[output_idx])

        if self.RGBFD:
            data.update(rgb_fd = torch.cat(mats_input))

        return data

class DDFF12(Dataset):
    def __init__(self, root_dir, split='train', shuffle=False, scale=1, img_num=10, visible_img=10, recon_all=True, RGBFD=True, DPT=True, AIF=True):
        self.hdf5 = h5py.File(root_dir, 'r')
        self.shuffle = shuffle
        self.img_num = img_num
        self.scale = scale
        self.visible_img = visible_img
        self.disp_dist = torch.linspace(0.02,0.28, steps=10)
        self.focus_dist = torch.from_numpy(self.disp2dpt(self.disp_dist))
        self.recon_all = recon_all
        self.RGBFD = RGBFD
        self.DPT = DPT
        self.AIF = AIF
        
        ##### Load and sort all images
        self.imglist_all = self.hdf5[f'stack_{split}']
        self.imglist_disp = self.hdf5[f'disp_{split}']
        #self.imglist_aif = self.hdf5[f'AiF_{split}']

        self.n_stack = self.imglist_all.shape[0]
        print(f"Total number of samples: {self.n_stack}\n{self.visible_img} out of {self.img_num} images per sample are visible for input")

    def __getitem__(self, idx):
        ##### Read and process an image
        img_idx = idx

        sub_idx = np.arange(self.img_num)
        if self.shuffle:
            np.random.shuffle(sub_idx)
        input_idx = sub_idx[:self.visible_img]
        if self.recon_all:
            output_idx = sub_idx
        else:
            output_idx = sub_idx[self.visible_img:]

        mats_input = []
        mats_output = []

        for i in sub_idx:
            mat_all = (self.imglist_all[idx][i]/255.).astype(np.float32)
            #if self.scale == 1:
            #    mat_all = cv2.pyrDown(mat_all)
            #if self.scale == 2:
            #    mat_all = cv2.pyrDown(mat_all)
            #    mat_all = cv2.resize(mat_all, (mat_all.shape[0]//2, mat_all.shape[1]//2))
            if i in output_idx:    
                mats_output.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))
            if self.RGBFD and i in input_idx:   
                mat_fd = self.focus_dist[i].view(-1, 1, 1).expand(*mat_all.shape[:-1], 1).numpy()
                mat_all = np.concatenate([mat_all, mat_fd], axis=-1)                 
                mats_input.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0))
        data = dict(output=torch.cat(mats_output), output_fd=self.focus_dist[output_idx])

        if self.RGBFD:
            data.update(rgb_fd = torch.cat(mats_input))

        if self.DPT:
            mat_dpt = self.disp2dpt(self.imglist_disp[idx])
            mat_dpt = torch.from_numpy(mat_dpt).unsqueeze(0)
            data.update(dpt = mat_dpt)

        #if self.AIF:
        #    mat_aif = (self.imglist_aif[idx]/255.).astype(np.float32)
        #    data.update(aif = torch.from_numpy(mat_aif.transpose(2, 0, 1)))

        return data
    
    def __len__(self):
        return self.n_stack
    
    def disp2dpt(self, disp):
        K2 = 1982.0250823695178
        flens = 7317.020641763665
        fsubaperture = 521.4052
        baseline = K2/flens*1e-3 # meters/ meter/px
        dpt = np.zeros_like(disp)
        dpt[disp!=0] = baseline*fsubaperture / disp[disp!=0]

        return dpt


# —————————————————————————————————————————————————— Defocus-Net ————————————————————————————————————————————————————————————
#dict(output, output_fd,    rgb_fd=[mat_all + fd], dpt, (aif))
from re import sub
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
import h5py
import cv2
import torchvision.transforms as T

class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""
    def __init__(self, root_dir, split='train', shuffle=False, img_num=5, focus_dist=[0.1,.15,.3,0.7,1.5], scale=1, RGBFD=True, DPT=True, AIF=False, max_dpt = 3.):
        self.root_dir = root_dir
        self.dpt_path = root_dir
        self.img_num = img_num
        self.focus_dist = torch.Tensor(focus_dist)
        self.shuffle = shuffle
        self.scale = scale
        # 暂时默认recon_all
        self.near = 0.1
        self.far = 3.0
        self.RGBFD = RGBFD
        self.DPT = DPT
        self.AIF = AIF

        ##### Load and sort all images
        self.imglist_all = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f[-7:] == "Dpt.png"]

        print("Total number of samples", len(self.imglist_dpt), "  Total number of seqs", len(self.imglist_dpt) / img_num)

        self.imglist_all.sort()
        self.imglist_dpt.sort()

        #self.camera = CameraLens(2.9 * 1e-3, f_number=f_number)
        self.max_dpt = max_dpt

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        ##### Read and process an image
        img_idx = idx
        sub_idx = np.arange(self.img_num)
        #if self.shuffle:
        #    np.random.shuffle(sub_idx)

        idx_dpt = int(idx)


        ind = idx * self.img_num

        # add RGB, CoC, Depth inputs
        mats_input = []
        mats_output = []

        for i in sub_idx:
            img_all = cv2.imread(self.root_dir + self.imglist_all[img_idx + i])
            img_all = np.array(img_all)
            #if self.scale == 1:
            #    img_all = cv2.pyrDown(img_all)
            #if self.scale == 2:
            #    img_all = cv2.pyrDown(img_all)
            #    img_all = cv2.pyrDown(img_all)
            mat_all = img_all.copy() / 255.
            mats_output.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0).float())

            if self.RGBFD: 
                mat_fd = self.focus_dist[i].view(-1, 1, 1).expand(*mat_all.shape[:-1], 1).numpy()
                mat_all = np.concatenate([mat_all, mat_fd], axis=2)
                mats_input.append(torch.from_numpy(mat_all.transpose((2, 0, 1))).unsqueeze(0).float())
        data = dict(output=torch.cat(mats_output), output_fd=self.focus_dist[sub_idx])

        if self.RGBFD:
            data.update(rgb_fd = torch.cat(mats_input))

        if self.DPT:
            img_dpt = Image.open(os.path.join(self.dpt_path, self.imglist_dpt[idx]))
            img_dpt = np.asarray(img_dpt, dtype=np.float32)
            img_dpt = np.clip(img_dpt / 1e2, self.near, self.far)
            H, W = img_dpt.shape
            img_dpt = cv2.resize(img_dpt, (W//self.scale, H//self.scale))
            mat_dpt = img_dpt.copy()
            mat_dpt = torch.from_numpy(mat_dpt).unsqueeze(0)
            data.update(dpt = mat_dpt)

        return data



