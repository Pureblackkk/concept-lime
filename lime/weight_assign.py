import clip
import torch
import json
import os
import pandas as pd
import numpy as np
import skimage.io as io
from PIL import Image
from tqdm import tqdm
from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity


CHUNK_SIZE = 2000
LOCAL_AVG_JSON_PATH = './lime/avg_weights.json'

class ImgDistanceWeight(ABC):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        
    def __chunk_list(self, l, chunk_size):
        return [l[i:i+chunk_size] for i in range(0, len(l), chunk_size)]

    def _save_avg_embedding(self, dataset_name, embeddings):
        if not os.path.exists(LOCAL_AVG_JSON_PATH):
            print(f"weights file is not existed, created")
            with open(LOCAL_AVG_JSON_PATH, "w") as f:
                data = {}
                data[dataset_name] = embeddings
                json.dump(data, f)
                print('Avg embedding has been written')
        else:
            with open(LOCAL_AVG_JSON_PATH, "r") as f:
                data = json.load(f)
            with open(LOCAL_AVG_JSON_PATH, "w") as f:
                data[dataset_name] = embeddings
                json.dump(data, f)
                print('Avg embedding has been written')
           

    def _cal_avg_embedding(self, img_path_list):
        # Load img from img list
        images_path_chunked = self.__chunk_list(img_path_list, CHUNK_SIZE)
        embedding_list = []

        for image_paths in tqdm(images_path_chunked):
            image_list = [Image.open(image).convert('RGB') for image in image_paths]
            image_inputs = torch.cat([
                self.preprocess(pil_image).unsqueeze(0) for pil_image in image_list
            ]).to(self.device)

            # Get embeddings
            with torch.no_grad():
                img_embedding = self.model.encode_image(image_inputs)
            
            embedding_list.append(img_embedding)
        
        # Calculate the avarge
        concat_embeddings = torch.cat(embedding_list, dim=0)
        avg_embedding = torch.mean(concat_embeddings, dim=0)

        # Set to public property
        self.avg_embedding = avg_embedding

        return avg_embedding.tolist()
    
    def _check_local_avg_exist(self, dataset_name: str):
        '''
            Check if local avg weights exist.
            If it is, load it directly(do not need to calculate for each time)
        '''
        try:
            with open(LOCAL_AVG_JSON_PATH, 'r') as f:
                local_avg_json = json.load(f)
                if dataset_name in local_avg_json:
                    # Avg exists, then load it directly
                    self.avg_embedding = np.array(local_avg_json[dataset_name])
                    return True
                
                return False
        except Exception as e:
            print(e)
            return False
    
    def cosin_distance(self, img):
        img = Image.open(img).convert('RGB')
        img = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(img).cpu().numpy()[0]
        
        cos_dis = 1 - cosine_similarity([embedding], [self.avg_embedding])[0][0]

        return cos_dis
    
    @abstractmethod
    def get_name(self):
        pass

class CelebAImgDistanceWeight(ImgDistanceWeight):
    def __init__(
        self,
        dataset_name: str,
        class_name_id_pair: Tuple[str, int],
        meta_data_path: str,
        img_src_dir: str,
    ):
        super().__init__()
        class_name, id_num = class_name_id_pair
        id_num = None if (type(id_num)==str and  len(id_num) == 0) else id_num
        self.class_name = class_name
        self.dataset_name = f'CelebA-{self.class_name}'

        # Check local avg exists
        if not super()._check_local_avg_exist(self.dataset_name):
            # If not Init avg embedding
            meta_img_list = self.__init_meta_img_list(
                meta_data_path,
                img_src_dir,
                id_num
            )
            avg_embedding = super()._cal_avg_embedding(meta_img_list)

            # Save to the json file
            super()._save_avg_embedding(self.dataset_name, avg_embedding)
    
    def __init_meta_img_list(
        self,
        meta_data: str,
        img_src_dir: str,
        id_num: int = None
    ):
        # Load meta file dataframe
        df = pd.read_csv(
            meta_data,
            sep=r'\s+',
            header=0
        )
        df = df.reset_index()
        df.columns.values[0] = 'image_id'

        # Get corresponding file list
        if not id_num is None:
            # TODO: Fit celebA dataset, not implemented
            # TODO: currently using 'all'
            file_name_list = df.loc[df["y"] == id_num, "img_filename"].tolist()
        else:
            file_name_list = df["image_id"].tolist()
        file_path_list = [str(os.path.join(img_src_dir, img_name)) for img_name in file_name_list]

        return file_path_list

    def get_name(self):
        return f'cos_distance_weight_from_{self.class_name}'

class WaterBirdImgDistanceWeight(ImgDistanceWeight):
    def __init__(
        self,
        dataset_name: str,
        class_name_id_pair: Tuple[str, int],
        meta_data_path: str,
        img_src_dir: str,
    ):
        super().__init__()
        class_name, id_num = class_name_id_pair
        self.class_name = class_name
        self.dataset_name = f'Waterbird-{self.class_name}'

        # Check local avg exists
        if not super()._check_local_avg_exist(self.dataset_name):       
            # If not Init avg embedding
            meta_img_list = self.__init_meta_img_list(
                meta_data_path,
                img_src_dir,
                id_num
            )
            avg_embedding = super()._cal_avg_embedding(meta_img_list)

            # Save to the json file
            super()._save_avg_embedding(self.dataset_name, avg_embedding)
    
    def __init_meta_img_list(
        self,
        meta_data: str,
        img_src_dir: str,
        id_num: int = None
    ):
        df = pd.read_csv(meta_data)
        # Get corresponding file list
        if not id_num is None:
            file_name_list = df.loc[df["y"] == id_num, "img_filename"].tolist()
        else:
            file_name_list = df["img_filename"].tolist()
        file_path_list = [str(os.path.join(img_src_dir, img_name)) for img_name in file_name_list]

        return file_path_list
    
    def get_name(self):
        return f'cos_distance_weight_from_{self.class_name}'


if __name__ == '__main__':
    # test_waterbird_weight_assign = WaterBirdImgDistanceWeight(
    #     'WaterBird',
    #     ('all', None),
    #     '/home/pureblackkkk/b2t-master/data/cub/data/waterbird_complete95_forest2water2/metadata.csv',
    #     '/home/pureblackkkk/b2t-master/data/cub/data/waterbird_complete95_forest2water2'
    # )

    # test_waterbird_weight_assign.cosin_distance(
    #     '/home/pureblackkkk/b2t-master/data/cub/data/waterbird_complete95_forest2water2/002.Laysan_Albatross/Laysan_Albatross_0012_696.jpg'
    # )

    # test_celebA_weight_assign = CelebAImgDistanceWeight(
    #     'CelebA',
    #     ('all', None),
    #     '/home/pureblackkkk/my_volume/celebA/CelebA/Anno/list_attr_celeba.txt',
    #     '/home/pureblackkkk/b2t-master/data/celebA/data/img_align_celeba'
    # )

    # print(test_celebA_weight_assign.cosin_distance(
    #     '/home/pureblackkkk/my_volume/concept-lime-data/revert_celebA_equal_2000_Feb_7/celeba_best_model_CelebA_erm_blond_without_class/3.png'
    # ))
    pass