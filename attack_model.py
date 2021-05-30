#FIXME ..or delete

# import cv2
import math
import os
import sys
import time
import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler, sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

import common.datasets
import common.eval
import common.paths
import common.state
import common.utils
import attacks
from common.log import log
from train_utils import *

warnings.filterwarnings("ignore")

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')

batch_size = 12


model_file = r'D:\Projects\Research\coloradvtr\assets\models\classifier.pth.tar'
state = common.state.State.load(model_file)
model = state.model
model = model.cuda()
model.eval()

data_dir_train = r'D:\Projects\Research\Snapshots'

image_transforms = {
    #     # Train uses data augmentation
        'train':
        transforms.Compose([
            # transforms.RandomResizedCrop(size=299),#, scale=(1., 1.0)
            transforms.RandomPerspective(distortion_scale=0.45, p=0.4),
            # transforms.RandomRotation(degrees=15),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(size=400),
            transforms.Resize(size=224),
            transforms.ColorJitter(0.3, 0.35, 0.3, 0.04),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    #     # Validation does not use augmentation
        'val':
        transforms.Compose([
            # transforms.Resize(size=(350)),
            # transforms.CenterCrop(299),
            transforms.CenterCrop(size=400),
            transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
dataloaders = get_dataloaders(data_dir_train, 0.3,  batch_size, image_transforms)
trainloader = dataloaders['train']
testloader = dataloaders['val']
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size-6, shuffle=False, num_workers=0)

epsilon = 0.03
ace = attacks.AdvCF()
ace.num_classes = 2
ace.search_steps = 1
ace.max_iterations = 150
ace.initial_const = 10
ace.device = torch.device("cuda")
ace.pieces = 64

torch.manual_seed(12)
torch.backends.cudnn.deterministic = True

output_path = ('./.assets/data/AdvCF_conv')
os.makedirs(output_path, exist_ok=True)

classes = ['safe', 'bomb']
num_classes = 2

namer = 0
for data, ori_labels in tqdm(trainloader, desc="Gen_adv"):
    data = data.cuda()
    ori_labels = ori_labels.cuda()
    # target_labels = torch.zeros(len(ori_labels), dtype=torch.long).to(device)
    #Apply our AdvCF to genrate the adversarial images
    X_adv,o_best_l2 = ace.adversary(model, data, ori_labels, targeted=False)
    # X_adv,o_best_l2 = approach.adversary(model, data, target_labels, pieces=pieces, targeted=True)

    #save the successfully adversarial images
    for j, adv_img in enumerate(X_adv):
        if o_best_l2[j]<1e9:
            cur_output_path = os.path.join(output_path, classes[ori_labels[j]]+'/')
            if not os.path.exists(cur_output_path):
                os.makedirs(cur_output_path)
            x_np=transforms.ToPILImage()(adv_img.detach().cpu())
            x_np.save(os.path.join(cur_output_path, 'adv_'+str(namer)+'.png'))
            namer += 1
