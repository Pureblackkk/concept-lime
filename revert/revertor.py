import os
import json
import subprocess
import pandas as pd
from typing import List

TEMP_JSON_PATH = './tmp.json'
IMG2TEXT_SCRIPT_DIR = './revert/'
IMG2TEXT_SCRIPT_NAME = 'text2image.py'

class Revertor:
    def __init__(
        self,
        src_dir: str,
        save_dir: str,
        model_path_or_name: str,
    ):
        self.src_dir = src_dir
        self.save_dir = save_dir
        self.model_path_or_name = model_path_or_name
    
    def __load_src_keyword(self):
        file_list = os.listdir(self.src_dir)
        return file_list
    
    def __create_tmp_json_file(self):
        with open(TEMP_JSON_PATH, 'w') as f:
            json.dump({}, f, indent=4)

    def __save2tmp_json_file(self, pair_dict):
        with open(TEMP_JSON_PATH, 'w') as f:
            json.dump(pair_dict, f, indent=4)

    def __delete_tmp_json_file(self):
        os.remove(TEMP_JSON_PATH)
    
    def __text2image(
        self,
        prompt_list: List[str],
        without_class: bool,
        file_name: str,
    ):
        file_name = file_name.split('.')[0]
        # Create directory for the data to be saved
        class_suffix = '_without_class' if without_class else '_with_class'
        saved_dir_name = f'{file_name}{class_suffix}'
        saved_final_dir = os.path.join(self.save_dir, saved_dir_name)

        if not os.path.exists(saved_final_dir):
            os.makedirs(saved_final_dir)
            print(f'created path: {saved_final_dir}')

        pair_list = []
        for idx, prompt in enumerate(prompt_list):
            # Create (prompt, save_path) pairs to be generated
            pair_list.append([prompt, f'{saved_final_dir}/{idx}.png'])
        
        return pair_list
            
    def revert(self, **kwargs):
        '''
            Revert prompt to image
        '''
        # Use the prompt with class or not
        without_class = kwargs.get('without_class', True)
        num_process = kwargs.get('num_process', 1)

        # Is extend mode
        extend_mode = kwargs.get('extend_mode', False)
        extend_mode_command = '--extend_mode' if extend_mode else ''

        # Create temp json file
        self.__create_tmp_json_file()
        prompt_pair = {'prompt_pair': []}

        file_list = self.__load_src_keyword()

        for file_name in file_list:
            df = pd.read_csv(os.path.join(self.src_dir, file_name))
            promptList = None

            if without_class:
                promptList = df['prompt_without_class']
            else:
                promptList = df['prompt_with_class']
        
            pair_list = self.__text2image(
                promptList,
                without_class,
                file_name,
            )

            current_prompt_pair = prompt_pair['prompt_pair']
            current_prompt_pair.extend(pair_list)
            prompt_pair['prompt_pair'] = current_prompt_pair

        # Saving the prompt_pair
        self.__save2tmp_json_file(prompt_pair)

        # Start text2img program
        result = subprocess.run([
            'accelerate', 'launch',
            f'{os.path.join(IMG2TEXT_SCRIPT_DIR, IMG2TEXT_SCRIPT_NAME)}',
            '--model_name_or_path', self.model_path_or_name,
            '--temp_json_path', TEMP_JSON_PATH,
            f'{extend_mode_command}',
            '--num_processes', str(num_process),
        ])

        if result.returncode == 0:
            print("Text to image generation done")
            # Remove temp file
            self.__delete_tmp_json_file()

        else:
            print(f"Text to image exited with return code {result.returncode}")
            

