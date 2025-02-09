import os
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List
from prompt_generator import PromptGenerator

# Enable tqdm for pandas
tqdm.pandas()

class DataDisturbing:
    def __init__(
        self,
        src_dir: str,
        save_dir: str,
        model_name: str,
    ):
        self.src_dir = src_dir
        self.save_dir = save_dir
        self.prompt_generator = PromptGenerator(model_name)
    
    def __load_src_keyword(self):
        file_list = os.listdir(self.src_dir)
        return file_list
    
    def __save_target_keyword(self, name, df):
        os.makedirs(self.save_dir, exist_ok=True)
        df.to_csv(os.path.join(self.save_dir, name), index=False)

    def __generate_prompt(self, row, class_name:str = None, **kwargs):
        present_keywords = [col for col, val in row.items() if val == 1]

        # Add class name if provided
        if class_name:
            present_keywords.append(class_name)

        template_type = kwargs.get('template_type')
        shots_num = kwargs.get('shots_num')

        return self.prompt_generator.generate_prompt(
            present_keywords,
            template_type,
            shots_num,
        )

    def __disturbing_single_class_keyword_with_prob(
        self,
        keywords: List[str],
        class_name: str,
        file_name: str,
        **kwargs,
    ):
        keyword_matrix = np.random.randint(0, 2, size=(kwargs.get('sample_num'), len(keywords)))
        df = pd.DataFrame(keyword_matrix, columns=keywords)

        if kwargs.get('generate_prompt', True):
            # Aggregate present keyword for each row and generate template by GPT
            df['prompt_without_class'] = df.progress_apply(
                lambda row: self.__generate_prompt(row, None, **kwargs),
                axis=1
            )
            
            # Add with class when needed
            if kwargs.get("prompt_with_class", False):
                df['prompt_with_class'] = df.progress_apply(lambda row: self.__generate_prompt(row, class_name, **kwargs), axis=1)

        # Save the dataframe
        self.__save_target_keyword(file_name, df)
    
    def __disturbing_single_class_keyword_with_equal(
        self,
        keywords: List[str],
        class_name: str,
        file_name: str,
        **kwargs,
    ):  
        '''
        For each keyword, the total probability would be 0.5
        '''
        sample_num = kwargs.get('sample_num')

        # Create dataframe
        df = pd.DataFrame()

        for keyword in keywords:
            values = [i % 2 for i in range(sample_num)]
            random.shuffle(values)
            df[keyword] = values
        
        if kwargs.get('generate_prompt', True):
            # Aggregate present keyword for each row and generate template by GPT
            df['prompt_without_class'] = df.progress_apply(
                lambda row: self.__generate_prompt(row, None, **kwargs),
                axis=1
            )
            # Add with class when needed
            if kwargs.get("prompt_with_class", False):
                df['prompt_with_class'] = df.progress_apply(lambda row: self.__generate_prompt(row, class_name, **kwargs), axis=1)

        # Save the dataframe
        self.__save_target_keyword(file_name, df)
        
    def disturbing(self, **kwargs):
        '''
            Disturbing keyword and with generated captions prompt
        '''
        file_list = self.__load_src_keyword()

        for file_name in file_list:
            df = pd.read_csv(os.path.join(self.src_dir, file_name))

            class_name = file_name.split('_')[-1].split('.')[0]
            keywords = df['Keyword'].tolist()

            self.__disturbing_single_class_keyword_with_equal(
                keywords,
                class_name,
                file_name,
                **kwargs
            )