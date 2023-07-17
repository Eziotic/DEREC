import sys
sys.path.append('..')

import numpy as np
from tqdm.auto import tqdm, trange
# from torch.utils.tensorboard import SummaryWriter
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from model import *
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Training Config')
parser.add_argument('--name', '-N', type=str, required=True)

# Data
parser.add_argument('--data_path', type=str, default="/mnt/cfs/sihaozhe/data/fs_7_05")
parser.add_argument('--dataset', type=str, choices=['render', 'NYUv2'], default='render')
parser.add_argument('--normalize_dpt', action='store_true', default=False)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--image_num', type=int, default=5)
parser.add_argument('--visible_image_num', type=int, default=5)
parser.add_argument('--recon_all', type=bool, default=True)
parser.add_argument('--RGBFD', type=bool, default=True)
parser.add_argument('--DPT', type=bool, default=True)
parser.add_argument('--AIF', type=bool, default=True)

# Train
parser.add_argument('--gt_dpt', action='store_true', default=False)
parser.add_argument('--gt_aif', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--BS', '-B', type=int, default=32)
parser.add_argument('--epoch', '-E', type=int, default=2000)
parser.add_argument('-W', type=int, default=16)
parser.add_argument('-D', type=int, default=4)

parser.add_argument('--blur_loss_lambda', type=float, default=1e-1)
parser.add_argument('--recon_loss_lambda', type=float, default=1e3)
parser.add_argument('--sm_loss_lambda', type=float, default=1e1)
parser.add_argument('--sharp_loss_lambda', type=float, default=1e2)
parser.add_argument('--recon_loss_alpha', type=float, default=0.85)
parser.add_argument('--sm_loss_beta', type=float, default=1.)
parser.add_argument('--blur_loss_sigma', type=float, default=1.)
parser.add_argument('--blur_loss_window', type=int, default=7)

parser.add_argument('--aif_recon_loss_alpha', type=float, default=0.85)
parser.add_argument('--aif_recon_loss_lambda', type=float, default=1e1)
parser.add_argument('--aif_blur_loss_lambda', type=float, default=1)

parser.add_argument('--dpt_post_op', type=str, choices=['raw', 'clip', 'norm'], default='raw')

# Render 
parser.add_argument('--window_size', type=int, default=7)
parser.add_argument('--soft_merge', type=bool, default=False)

# Camera
parser.add_argument('--fnumber', type=float, default=0.5)
parser.add_argument('--focal_length', type=float, default=2.9*1e-3)
parser.add_argument('--sensor_size', type=float, default=3.1*1e-3)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--camera_near', type=float, default=0.1)
parser.add_argument('--camera_far', type=float, default=1.)

# Logging
parser.add_argument('--log', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--VIS_FREQ', type=int, default=100)

# Saving
parser.add_argument('--SAVE_FREQ', type=int, default=5000)
parser.add_argument('--save_best', action='store_true', default=False)
parser.add_argument('--save_last', action='store_true', default=False)
parser.add_argument('--save_checkpoint', action='store_true', default=False)

# Misc
parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--TEST_FREQ', type=int, default=500)
parser.add_argument('--verbose', action='store_true', default=False)

args = parser.parse_args()
name = ('_').join([args.name, f'rec{args.recon_loss_lambda}', f'sharp{args.sharp_loss_lambda}', f'sm{args.sm_loss_lambda}', f'blur{args.blur_loss_lambda}',
            f'E{args.epoch}', f'B{args.BS}', f'W{args.W}', f'D{args.D}']+['soft' if args.soft_merge else 'hard'])
print(f'Running Experiment {name}')

if args.use_cuda:
    torch.cuda.set_device(args.device)
    print(f"Running on CUDA{args.device}")
else:
    args.device = torch.device("cpu")
    print(f"Running on CPU")

if args.log:
    wandb.init(project="SS-DFD", dir='..', name=name)
    wandb.config.update(args)

save_path = os.path.join('../exp/', name)
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_savepath = os.path.join(save_path, 'model')
if not os.path.exists(model_savepath):
    os.mkdir(model_savepath)

dataset_config = get_data_config(args)
train_dl, test_dl, _ = load_data(dataset_config, args.dataset, args.BS)

def train(model, render, criterion, optimizer, scheduler, n_iter, train_dl, test_dl, args):
    min_test_loss = np.inf
    if args.verbose:
        iter_ = trange(args.epoch, dynamic_ncols=True, unit='Epoch')
    else:
        iter_ = range(args.epoch)
    for e in iter_:
        for i, batch in enumerate(train_dl):
            if args.use_cuda:
                batch['rgb_fd'] = batch['rgb_fd'].cuda()
                batch['output'] = batch['output'].cuda()
                batch['output_fd'] = batch['output_fd'].cuda()
                batch['dpt'] = batch['dpt'].cuda()   
                batch['aif'] = batch['aif'].cuda()   
            
            B, FS, C, H, W = batch['output'].shape
            aif_d = model(batch['rgb_fd'])
            
            aif = batch['aif'] if args.gt_aif else aif_d[:, :-1]
            dpt = batch['dpt'] if args.gt_dpt else aif_d[:, -1]
            dpt = dpt_post_op(dpt, args)

            fs_aif = aif.unsqueeze(1).expand(B, FS, C, H, W).contiguous().view(B*FS, C, H, W)
            fs_dpt = dpt.unsqueeze(1).expand(B, FS, H, W).contiguous().view(B*FS, H, W)
            fd = batch['output_fd'].view(-1, 1, 1).expand_as(fs_dpt)
            defocus = camera.getCoC(fs_dpt, fd)
            recon = render(fs_aif, defocus)

            grey_fs = torch.mean(batch['rgb_fd'][:, :, :-1], dim=-3)
            if args.soft_merge:
                clear_w = F.softmin(defocus.view(B, FS, H, W), dim=1)
                coarse_aif = torch.sum(grey_fs * clear_w, dim=1, keepdim=True)
            else:
                clear_idx = torch.argmin(defocus.view(B, FS, H, W), dim=1, keepdim=True)
                coarse_aif = torch.gather(grey_fs, 1, clear_idx)

            loss_ssim = 1 - criterion['ssim'](recon, batch['output'].view(B*FS, 3, H, W))
            loss_l1 = criterion['l1'](recon, batch['output'].view(B*FS, 3, H, W))
            loss_sharp = criterion['sharp'](recon, batch['output'].view(B*FS, 3, H, W))
            loss_recon = args.recon_loss_alpha * loss_ssim + (1 - args.recon_loss_alpha) * loss_l1
            loss = loss_recon * args.recon_loss_lambda + loss_sharp * args.sharp_loss_lambda 

            if not args.gt_aif:
                loss_aif_ssim = 1 - criterion['ssim'](aif, coarse_aif)
                loss_aif_l1 = criterion['l1'](aif, coarse_aif)
                loss_aif_recon = args.aif_recon_loss_alpha * loss_aif_ssim + (1 - args.aif_recon_loss_alpha) * loss_aif_l1
                loss_aif_blur = criterion['blur'](aif)
                loss += args.aif_recon_loss_lambda * loss_aif_recon + args.aif_blur_loss_lambda * loss_aif_blur

            if not args.gt_dpt:
                loss_blur = criterion['blur'](coarse_aif)
                loss_dpt = criterion['l1'](dpt.unsqueeze(1), batch['dpt']) 
                loss_smooth = criterion['smooth'](dpt, aif) 
                loss = loss + loss_smooth * args.sm_loss_lambda + loss_blur * args.blur_loss_lambda 

            if args.log:
                logs = dict(ssim=1-loss_ssim.item(), l1=loss_l1.item(), sharp=loss_sharp.item(), blur=loss_blur.item(), 
                            recon=loss_recon.item(), dpt=loss_dpt.item(), loss=loss.item(), smooth=loss_smooth.item())
                wandb.log({'train':logs})

            if n_iter % args.SAVE_FREQ == 0 and args.save_checkpoint:
                state = {
                    'iter': n_iter,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(model_savepath, f'checkpoint-{n_iter}.pth'))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            n_iter += 1

        if e % args.VIS_FREQ == 0 and args.vis:
            log_img = True
        else:
            log_img = False

        test_loss = test(model, render, criterion, test_dl, log_img, args)
        if args.save_best and min_test_loss > test_loss:
            state = {
                'iter': n_iter,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(model_savepath, f'best-model.pth'))
    return n_iter

def test(model, render, criterion, dataloader, log_img, args):
    losses = []
    losses_dpt = []
    n_test = len(dataloader)

    if args.log and log_img:
        rand_img_idx = np.random.randint(0, n_test)
    else:
        rand_img_idx = -1

    for i, batch in enumerate(dataloader):
        if i == n_test:
            break
        if args.use_cuda:
            batch['rgb_fd'] = batch['rgb_fd'].cuda()
            batch['output'] = batch['output'].cuda()
            batch['output_fd'] = batch['output_fd'].cuda()
            batch['dpt'] = batch['dpt'].cuda()   
            batch['aif'] = batch['aif'].cuda()   
        
        with torch.no_grad():
            B, FS, C, H, W = batch['output'].shape
            aif_d = model(batch['rgb_fd'])
            aif = batch['aif']
            dpt = aif_d[:, -1]
            dpt = dpt_post_op(dpt, args)
            fs_aif = aif.unsqueeze(1).expand(B, FS, C, H, W).contiguous().view(B*FS, C, H, W) # use gt aif to verify model
            fs_dpt = dpt.unsqueeze(1).expand(B, FS, H, W).contiguous().view(B*FS, H, W)
            fd = batch['output_fd'].view(-1, 1, 1).expand_as(fs_dpt)
            defocus = camera.getCoC(fs_dpt, fd)
            recon = render(fs_aif, defocus)

            grey_fs = torch.mean(batch['rgb_fd'][:, :, :-1], dim=-3)
            if args.soft_merge:
                clear_w = F.softmin(defocus.view(B, FS, H, W), dim=1)
                coarse_aif = torch.sum(grey_fs * clear_w, dim=1, keepdim=True)
            else:
                clear_idx = torch.argmin(defocus.view(B, FS, H, W), dim=1, keepdim=True)
                coarse_aif = torch.gather(grey_fs, 1, clear_idx)

            if i == rand_img_idx:
                # aif_gt = batch['aif'].squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                # aif_gt_wandb = wandb.Image(aif_gt, caption=f'AIF GT')
                defocus_gt = []
                defocus_recon = []
                for i in range(batch['output'].shape[1]):
                    recon_def = recon[i].detach().cpu().numpy().transpose(1, 2, 0)
                    recon_wandb = wandb.Image(recon_def, caption=f"Recon Defocus, fd={batch['output_fd'][0, i]}")
                    defocus_recon.append(recon_wandb)
                    gt_def = batch['output'][0, i].detach().cpu().numpy().transpose(1, 2, 0)
                    gt_wandb = wandb.Image(gt_def, caption=f"GT Defocus, fd={batch['output_fd'][0, i]}")
                    defocus_gt.append(gt_wandb)
                recon_dpt = dpt[0].unsqueeze(-1).squeeze().detach().cpu().numpy()
                recon_dpt_wandb = wandb.Image(np.clip(recon_dpt/(np.max(recon_dpt)+1e-8), 0, args.camera_far)/args.camera_far, caption='Recon Dpt')
                gt_dpt = batch['dpt'][0].unsqueeze(-1).squeeze().detach().cpu().numpy()
                gt_dpt_wandb = wandb.Image(gt_dpt/args.camera_far, caption='GT Dpt')
                wandb.log(dict(defocus_gt = defocus_gt, defocus_recon=defocus_recon, gt_dpt=gt_dpt_wandb, recon_dpt=recon_dpt_wandb))

            loss_ssim = 1 - criterion['ssim'](recon, batch['output'].view(B*FS, 3, H, W))
            loss_l1 = criterion['l1'](recon, batch['output'].view(B*FS, 3, H, W))
            loss_smooth = criterion['smooth'](dpt, batch['aif'])
            loss_sharp = criterion['sharp'](recon, batch['output'].view(B*FS, 3, H, W))
            loss_recon = args.recon_loss_alpha * loss_ssim + (1 - args.recon_loss_alpha) * loss_l1
            loss_dpt = criterion['l1'](dpt.unsqueeze(1), batch['dpt'])
            loss_blur = criterion['blur'](coarse_aif)
            loss = loss_smooth * args.sm_loss_lambda + loss_sharp * args.sharp_loss_lambda + loss_recon * args.recon_loss_lambda + loss_blur * args.blur_loss_lambda

            losses.append(loss.item())
            losses_dpt.append(loss_dpt.item())
            
    if args.log:
        logs = dict(loss=np.mean(losses), dpt_loss=np.mean(losses_dpt))
        wandb.log({'test':logs})
    return np.mean(losses)

model = FUNet(4, 4, args.W, args.D)
render = GaussPSF(args.window_size)
if args.use_cuda:
    model.cuda()
    render.cuda()

optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dl)*args.epoch)
criterion = dict(l1=BlurMetric('l1'), smooth=BlurMetric('smooth', beta=args.sm_loss_beta), sharp=BlurMetric('sharp'), 
                ssim=BlurMetric('ssim'), blur=BlurMetric('blur', sigma=args.blur_loss_sigma, kernel_size=args.blur_loss_window))
camera = get_camera(args)

n_iter = 0
n_iter = train(model, render, criterion, optimizer, scheduler, n_iter, train_dl, test_dl, args)

if args.save_last:
    state = {
        'model': model.state_dict(),
        'iter': n_iter,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(model_savepath, f'final-model.pth'))
