import sys
sys.path.append('..')

import numpy as np
from tqdm.auto import tqdm, trange
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from model import *
from utils import *
from trainer import Trainer

parser = get_parser()
parser.add_argument('--n_shot', type=int, default=-1)
parser.add_argument('--load_model', type=str, default='NYU100/model/best-model.pth')
parser.add_argument('--n_shot_indices', nargs='*', type=int, default=None)
args = parser.parse_args()

print(f'Running Experiment {args.name}')

if args.use_cuda:
    torch.cuda.set_device(args.device)
    print(f"Running on CUDA{args.device}")
else:
    args.device = torch.device("cpu")
    print(f"Running on CPU")

if args.log:
    wandb.init(project="neword", dir='..', name=args.name)
    wandb.config.update(args)

dataset_config = get_data_config(args)
dataloaders = load_data(dataset_config, args.dataset, args.BS, args.n_shot, args.n_shot_indices)
model = DAIFNet(4, 4, args.W, args.D)
render = GaussPSF(args.window_size)
if args.use_cuda:
    model.cuda()
    render.cuda()
optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloaders[0])*args.epoch)

trainer = Trainer(dataloaders, model, render, optimizer, scheduler, args)
if args.load_model != '':
    trainer.load_checkpoint(args.continue_from, args.load_model)
elif args.continue_from != '':
    trainer.load_checkpoint(args.continue_from)

if not args.eval:
    trainer.train()
trainer.eval_model()

