import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
# from simclr import SimCLR
# from simclr.modules import NT_Xent, get_resnet
# from simclr.modules.transformations import TransformsSimCLR
# from simclr.modules.sync_batchnorm import convert_model

# from model import load_optimizer, save_model
# from utils import yaml_config_hook

import torchvision.transforms as transforms

from PIL import Image





def get_features(images, simclr_model, batch_size, device):    
    h_list = []
    z_list = []
    scaler = transforms.Resize((224, 224))
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    for first in range(0, len(images), batch_size):
        images_subset = images[first:first+batch_size]
#         a = [normalize(to_tensor(scaler(Image.fromarray(im, "RGB")))) for im in images_subset]
        a = [to_tensor(scaler(Image.fromarray(im, "RGB"))) for im in images_subset]
        images_subset = torch.stack(a).to(device)
        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(images_subset, images_subset)
        h = h.detach()
        z = z.detach()
        
        h_list.extend(h.cpu().detach().numpy())
        z_list.extend(z.cpu().detach().numpy())
        
    h_features = np.array(h_list)
    z_features = np.array(z_list)
    # print("z features shape {}".format(z_features.shape))
    print("h features shape {}".format(h_features.shape))
    return h_features, z_features
    





def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader