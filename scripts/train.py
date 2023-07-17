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
from trainer import Trainer

parser = get_parser()
args = parser.parse_args()

if args.manual_seed != None:
    print(f'Manual Seed: {args.manual_seed}')
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

print(f'Running Experiment {args.name}')

device_ids = [0, 1, 2, 3, 4]

if args.use_cuda:
    torch.cuda.set_device(args.device)
    print(f"Running on CUDA:0")
else:
    args.device = torch.device("cpu")
    print(f"Running on CPU")

if args.log:
    wandb.init(project="neword", dir='..', name=args.name)
    wandb.config.update(args)

dataset_config = get_data_config(args)
dataloaders = load_data(dataset_config, args.dataset, args.BS)
model = DAIFNet(4, 4, args.W, args.D, args.pred_param)
if args.pred_param:
    param_model = ParamNet(256, 1, args.W, args.D)
render = GaussPSF(args.window_size)

if args.use_cuda:
    #model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()
    render.cuda()
    
optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloaders[0])*args.epoch)

trainer = Trainer(dataloaders, model, render, optimizer, scheduler, args)
if args.continue_from != '':
    trainer.load_checkpoint(args.continue_from)

if not args.eval:
    trainer.train()
trainer.eval_model()

