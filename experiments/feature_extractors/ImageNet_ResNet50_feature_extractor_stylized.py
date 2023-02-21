
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
root_path_dog_sketch = "./imagenet_folder/stylized_imagenet/dog_sketch" # path to images stylized with dog sketch
root_path_picasso_dog = "./imagenet_folder/stylized_imagenet/picasso_dog" # path to images stylized with Picasso dog
root_path_woman_sketch = "./imagenet_folder/stylized_imagenet/woman_sketch" # path to images stylized with woman sketch
root_path_picasso_sp= "./imagenet_folder/stylized_imagenet/picasso_sp" # path to images stylized with Picasso self portrait


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

val_transform = transforms.Compose(
            [
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
    img2vec = Img2Vec(model="resnet50")# resnet-50
    # img2vec = Img2Vec() # resnet-18
    for images_batch, labels_batch in tqdm(dataloader):
        Z_batch = img2vec.get_vec(images_batch)
        Z_list.append(Z_batch)
        labels_list.append(labels_batch.numpy())
    Z = np.vstack(Z_list)
    labels = np.array(labels_list).flatten()
    print("Z features shape", Z.shape)
    return Z, labels



# generate dataloaders for different imagenet styles
dog_sketch_train_dataloader, dog_sketch_val_dataloader = create_dataloaders(root_path, root_path_dog_sketch)
picasso_dog_train_dataloader, picasso_dog_val_dataloader = create_dataloaders(root_path, root_path_picasso_dog)
woman_sketch_train_dataloader, woman_sketch_val_dataloader = create_dataloaders(root_path, root_path_woman_sketch)
picasso_sp_train_dataloader, picasso_sp_val_dataloader = create_dataloaders(root_path, root_path_picasso_sp)


# Extract image features 
print("### Extracting features using a pre-trained resnet 50 ###")
Z_train_dog_sketch, train_dog_sketch_labels = get_features(dog_sketch_train_dataloader)
Z_train_picasso_dog, train_picasso_dog_labels = get_features(picasso_dog_train_dataloader)
Z_train_woman_sketch, train_woman_sketch_labels = get_features(woman_sketch_train_dataloader)
Z_train_picasso_sp, train_picasso_sp_labels = get_features(picasso_sp_train_dataloader)

Z_test_dog_sketch, test_dog_sketch_labels = get_features(dog_sketch_val_dataloader)
Z_test_picasso_dog, test_picasso_dog_labels = get_features(picasso_dog_val_dataloader)
Z_test_woman_sketch, test_woman_sketch_labels = get_features(woman_sketch_val_dataloader)
Z_test_picasso_sp, test_picasso_sp_labels = get_features(picasso_sp_val_dataloader)


# # Save extracted features and image labels as numpy files
save('../data/Z_train_dog_sketch_imagenet_resnet.npy', Z_train_dog_sketch)
save('../data/Z_train_picasso_dog_imagenet_resnet.npy', Z_train_picasso_dog)
save('../data/Z_train_woman_sketch_imagenet_resnet.npy', Z_train_woman_sketch)
save('../data/Z_train_picasso_sp_resnet.npy', Z_train_picasso_sp)


save('../data/Z_test_dog_sketch_imagenet_resnet.npy', Z_test_dog_sketch)
save('../data/Z_test_picasso_dog_imagenet_resnet.npy', Z_test_picasso_dog)
save('../data/Z_test_woman_sketch_imagenet_resnet.npy', Z_test_woman_sketch)
save('../data/Z_test_picasso_sp_imagenet_resnet.npy', Z_test_picasso_sp)


save('../data/train_dog_sketch_labels_imagenet.npy', train_dog_sketch_labels)
save('../data/train_picasso_dog_labels_imagenet.npy', train_picasso_dog_labels)
save('../data/train_woman_sketch_labels_imagenet.npy', train_woman_sketch_labels)
save('../data/train_picasso_sp_labels_imagenet.npy', train_picasso_sp_labels)

save('../data/test_dog_sketch_labels_imagenet.npy', test_dog_sketch_labels)
save('../data/test_picasso_dog_labels_imagenet.npy', test_picasso_dog_labels)
save('../data/test_woman_sketch_labels_imagenet.npy', test_woman_sketch_labels)
save('../data/test_picasso_sp_labels_imagenet.npy', test_picasso_sp_labels)

