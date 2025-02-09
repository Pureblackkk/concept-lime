from tqdm import tqdm
from typing import Any
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(
        self,
        img_data_dir: str,
        csv_path: str,
        transform = None,
    ):
        self.data_dir = img_data_dir
        self.transform = transform
        self.csv_path = csv_path

        # Read file names form data_dir
        self.file_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_filename = os.path.join(self.data_dir, self.file_list[idx])
        orginal_img = Image.open(img_filename).convert('RGB')
        
        if not self.transform is None:
            img = self.transform(orginal_img)
        
        resized_original_img = orginal_img.resize((224, 224))
        return img, np.array(resized_original_img), self.csv_path, str(img_filename), self.file_list[idx], idx

def get_transformer(dataset_name: str, is_processing_revert: bool = True):
    if is_processing_revert:
        # Processing the revert image, the sd generated image size (1024 x 1024)
        match dataset_name:
            case 'WaterBird':
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            case 'CelebA':
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
            case _:
                raise ValueError('Wrong dataset name')
    else:
        match dataset_name:
            case 'WaterBird':
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            case 'CelebA':
                return transforms.Compose([
                    transforms.CenterCrop(178),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
            case _:
                raise ValueError('Wrong dataset name')

def create_img_data_loader(**kwargs):
    # Dataset params
    preprocess = get_transformer(kwargs.get('dataset_name'))
    img_data_dir = kwargs.get('img_data_dir')
    csv_path = kwargs.get('csv_path')

    # Dataloader params
    batch_size = kwargs.get('batch_size')
    num_workers = kwargs.get('num_workers')

    data_set = MyDataset(
        img_data_dir,
        csv_path,
        preprocess
    )

    return DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )

def load_classify_model(model_path: str):
    model = torch.load(model_path, map_location=DEVICE)
    model = model.to(DEVICE)
    return model

def classify_model_runner_and_save(
    model: Any,
    img_data_loader: Any,
    csv_path: str,
    without_class: bool,
    weight_assigner: Any,
    gradCAM: Any,
):
    '''
    Run the classifier on reverted image
    '''
    # evaluation mode
    model.eval()

    # Read the csv dataframe
    df = None
    with open(csv_path, 'r') as f:
        df = pd.read_csv(f)
    is_with_class =  'prompt_without_class' if without_class else 'prompt_with_class'
    pred_col_name = f'model_pred_{is_with_class}'

    # Get df column lenght
    length_col = df.shape[0]
    
    # Initialize number of classify result
    # eg. the output dimension would be like (n, 2)
    is_dict_initialized = False

    # Initialize weights column
    if not weight_assigner is None:
        weight_col_name = weight_assigner.get_name()
        df[weight_col_name] = np.nan

    # Run model to get result
    for (images, original_img, _, image_path, image_name, idx) in tqdm(img_data_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        class_num = outputs.shape[1]

        # Generate gradcam if needed
        if gradCAM:
            gradCAM.generate_cam(
                images,
                original_img,
                image_name,
                idx,
                outputs,
            )
            
        # Initialize the classify num dict
        if not is_dict_initialized:
            for i in range(class_num):
                df[f'{pred_col_name}_{i}'] = np.nan
            is_dict_initialized = True

        # Save the preds back to the corresponding csv file
        for i in range(len(outputs)):
            for j in range(class_num):
                df.loc[idx[i].item(), f'{pred_col_name}_{j}'] = outputs[i][j].item()

            # Calculate the assigned weights
            if not weight_assigner is None:
                path = image_path[i]
                weights = weight_assigner.cosin_distance(path)
                df.loc[idx[i].item(), f'{weight_col_name}'] = weights

    # Save back to the csv
    df.to_csv(str(csv_path), index=False)
    
            