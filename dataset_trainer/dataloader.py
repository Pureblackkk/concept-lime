import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Waterbird(Dataset):
    def __init__(self, data_dir: str, type_name: str = 'train'):
        class_name = ['waterbird', 'landbird']
        
        # File name
        self.img_name = [
            img
            for clsss in class_name
            for img in os.listdir(os.path.join(data_dir, type_name, clsss)) 
        ]

        # File path
        self.img_path_array = [
            os.path.join(data_dir, type_name, clsss, img)
            for clsss in class_name
            for img in os.listdir(os.path.join(data_dir, type_name, clsss)) 
        ]

        # y value
        self.target = torch.tensor([
            1 if clsss == 'waterbird' else 0
            for clsss in class_name
            for img in os.listdir(os.path.join(data_dir, type_name, clsss)) 
        ])

        # background value
        self.background = list(map(lambda x: 0 if x.split('_')[0] == 'land' else 1, self.img_name))

        # Is marked
        self.is_marked = list(map(lambda x: 0 if x.split('_')[1] == 'X' else 1, self.img_name))

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_path_array[idx]).convert('RGB')

        x = self.transform(img)
        y = self.target[idx]
        background = self.background[idx]
        is_marked = self.is_marked[idx]
        path = self.img_path_array[idx]

        return x, y, (background, is_marked), path

class Urbancars(Dataset):
    def __init__(self, data_dir: str, type_name: str = 'train'):
        class_name = ['country', 'urban']
        
        # File name
        self.img_name = [
            img
            for clsss in class_name
            for img in os.listdir(os.path.join(data_dir, type_name, clsss)) 
        ]

        # File path
        self.img_path_array = [
            os.path.join(data_dir, type_name, clsss, img)
            for clsss in class_name
            for img in os.listdir(os.path.join(data_dir, type_name, clsss)) 
        ]

        # Y value
        self.target = torch.tensor([
            1 if clsss == 'urban' else 0
            for clsss in class_name
            for img in os.listdir(os.path.join(data_dir, type_name, clsss)) 
        ])

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_path_array[idx]).convert('RGB')
        
        x = self.transform(img)
        y = self.target[idx]
        path = self.img_path_array[idx]

        return x, y, 0, path

def load_dataloader(dataset_name: str, data_dir: str, bs_train=128, bs_val=128, num_workers=8):
    match dataset_name:
        case 'waterbirds':
            train_set = Waterbird(data_dir, 'train')
            val_set = Waterbird(data_dir, 'val')
            test_set = Waterbird(data_dir, 'test')
        case 'urbancars':
            train_set = Urbancars(data_dir, 'train')
            val_set = Urbancars(data_dir, 'val')
            test_set = Urbancars(data_dir, 'test')
        case _:
            raise ValueError("Unsupported dataset")
    
    train_loader = DataLoader(train_set, batch_size=bs_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs_val, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader