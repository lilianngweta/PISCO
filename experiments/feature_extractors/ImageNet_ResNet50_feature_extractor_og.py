import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from numpy import save
from tqdm import tqdm

from imagenet_resnet_feature_extraction import Img2Vec
from imagenet_dataset import ImageNetDataset


batch_size = 128


root_path = "./imagenet_folder" # path to folder containing ImageNet class labels 
root_path_og = "./imagenet_folder/ILSVRC/Data/CLS-LOC" # path to original ImageNet images (images that are not stylized)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def create_dataloaders(root_path, root_path_style):
    train_dataset = ImageNetDataset(root_path,root_path_style, "train", train_transform)
    train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size, 
                # num_workers=8, 
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )

    val_dataset = ImageNetDataset(root_path, root_path_style, "val/val_images", val_transform)
    val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size, 
                # num_workers=0, 
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )
    return train_dataloader, val_dataloader



# Function to extract image features using resnet    
def get_features(dataloader):
    Z_list = []
    labels_list = []
    img2vec = Img2Vec(model="resnet50")#resnet-50
    # img2vec = Img2Vec() # resnet-18
    for images_batch, labels_batch in tqdm(dataloader):
        Z_batch = img2vec.get_vec(images_batch)
        Z_list.append(Z_batch)
        labels_list.append(labels_batch.numpy())
    Z = np.vstack(Z_list)
    labels = np.array(labels_list).flatten()
    print("Z features shape", Z.shape)
    return Z, labels

# generate dataloaders for original imagenet data
og_train_dataloader, og_val_dataloader = create_dataloaders(root_path, root_path_og)


# Extract image features 
print("### Extracting features using a pre-trained resnet 50 ###")
Z_train_og, train_og_labels = get_features(og_train_dataloader)
Z_test_og, test_og_labels = get_features(og_val_dataloader)


# Save extracted features and image labels as numpy files
save('../data/Z_train_og_imagenet_resnet.npy', Z_train_og)
save('../data/Z_test_og_imagenet_resnet.npy', Z_test_og)
save('../data/train_og_labels_imagenet.npy', train_og_labels)
save('../data/test_og_labels_imagenet.npy', test_og_labels)
