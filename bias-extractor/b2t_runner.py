import os
import yaml
import subprocess
from typing import Dict, List

BUILTIN_CONFIG_PATH = './built-in.yaml'
B2T_MAIN_PY = './b2t.py'

class B2TRunner:
    def __init__(self, b2t_proj_dir: str = None):
        if b2t_proj_dir is None:
            raise ValueError('B2T project directory is not given')
        
        # Set b2t project directory
        self.b2t_proj_dir = b2t_proj_dir

        # Initialize built in params settings
        self.built_in_param = self.__init_built_in_running_params()
    
    def __init_built_in_running_params(self) -> Dict[str, List[str]] | None:
        try:
            with open(BUILTIN_CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                print(data)

                return {
                    'dataset': data['dataset'], 
                    'model': data['model']
                }
        except:
            return None

    def __is_dataset_built_in(self, dataset: str = None) -> bool:
        if self.built_in_param is None:
            return False

        return (dataset in self.built_in_param['dataset'])
    
    def __is_model_built_in(self, model: str = None) -> bool:
        if self.built_in_param is None:
            return False
        
        return (model in self.built_in_param['model'])


    def run_b2t(self, **kwargs):
        '''
            Run B2T project with provided params
        '''
        dataset = kwargs.get('dataset', None)
        model = kwargs.get('model', None)
        extract_caption = kwargs.get('extract_caption', True)
        save_result = kwargs.get('save_result', True)
        cal_score = kwargs.get('cal_score', False)

        if dataset is None or model is None:
            raise ValueError('Model or Dataset must be given ')
        
        # Check if dataset and model is bulit-in
        # If it is not, then copy the dataset and model into corresponding location in B2T
        
        if not self.__is_dataset_built_in(dataset):
            # TODO: Implement necessary preparation
            pass
            
        if not self.__is_model_built_in(model):
            # TODO: Implement necessary preparation
            pass

        # Run B2T project with the given params
        result = subprocess.run([
            'python',
            os.path.join(self.b2t_proj_dir, B2T_MAIN_PY),
            '--dataset', dataset,
            '--model', model,
            '--extract_caption', str(extract_caption),
            '--save_result', str(save_result),
            '--cal_score', str(cal_score),
        ], cwd=self.b2t_proj_dir)

        if result.returncode == 0:
            print("B2T completed!")
        else:
            print(f"B2T exited with return code {result.returncode}")
