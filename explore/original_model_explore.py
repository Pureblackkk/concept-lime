from typing import List, Any
import pandas as pd
import os
import sys
import torch
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lime")))
from model_related import MyDataset, create_img_data_loader

CelebA_IMG_PATH = '/home/pureblackkkk/my_volume/celebA/CelebA/Img/img_align_celeba'
CelebA_META_INFO = '/home/pureblackkkk/my_volume/celebA/CelebA/Anno/list_attr_celeba.txt'
CelebA_MODEL_PATH = '/home/pureblackkkk/b2t-master/model/best_model_CelebA_erm.pth'

class OriginalModelRes:
    def __init__(self, dataset: str, filter_param: Any = None):
        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = self.__get_data_loader_by_dataset(
            dataset,
            filter_param,
        )
        self.model = self.__get_model_by_dataset(dataset)
    
    def __get_model_by_dataset(self, dataset):
        match dataset:
            case 'CelebA':
                model = torch.load(CelebA_MODEL_PATH, map_location=self.device)
                model = model.to(self.device)
                return model
            case 'Waterbird':
                raise ValueError('Not implemented yet')

    def __get_file_filter(self, dataset: str, filter_param: Any = None):
        match dataset:
            case 'CelebA':
                celeba_meta_df = pd.read_csv(
                    CelebA_META_INFO,
                    sep=r'\s+',
                    header=0
                )
                celeba_meta_df = celeba_meta_df.reset_index()
                celeba_meta_df.columns.values[0] = 'image_id'
                
                # Closure for image selection
                def file_filter(img_list: List[str]):
                    valid_df = celeba_meta_df

                    for keyword, val in filter_param:
                        valid_df = valid_df[valid_df[keyword] == val]

                    return valid_df['image_id'].values

                return file_filter

            case 'Waterbird':
                raise ValueError('Not implemented yet')

    def __get_data_loader_by_dataset(self, dataset: str, filter_param: Any = None):
        match dataset:
            case 'CelebA':
                file_filter = self.__get_file_filter(dataset, filter_param)

                return create_img_data_loader(
                    img_data_dir=CelebA_IMG_PATH,
                    csv_path='',
                    batch_size=10,
                    num_workers=4,
                    dataset_name=dataset,
                    is_processing_revert=False,
                    file_filter=file_filter,
                )

            case 'Waterbird':
                raise ValueError('Not implemented yet')

    def run_model_predict(self, correct_id: int):
        wrong_predict_num = 0
        total_num = 0

        for (images, _, meta, _, img_name, _) in tqdm(self.data_loader):
            images = images.to(self.device)
            outputs = self.model(images)
            pred = torch.argmax(outputs, dim=-1)
            wrong_predict_num += len(list(filter(lambda x: x != correct_id, pred)))
            total_num += len(outputs)

        print(f'Total sample num: {total_num}, Wrong sample num: {wrong_predict_num}')

if __name__ == '__main__':
    celebaARunner = OriginalModelRes(
        'CelebA',
        [('Male', 1), ('Blond_Hair', 1)]
    )
    celebaARunner.run_model_predict(1)
