import yaml
import numpy as np
from typing import List
from gpt4all import GPT4All

TEMPLATE_PATH = './disturb/template.yaml'

class PromptTemplate:
    def __init__(self):
        self.template = self.__load_template_config()

    def __load_template_config(self):
        try:
            with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                template = yaml.safe_load(f)
                return template
        except Exception as e:
            raise IOError(e)
    
    def __assmble_elemets(self, elements: List[str]):
        # Generate elements template
        elements_item_format = [f'{idx+1}. {element}' for idx, element in enumerate(elements)]

        return '\n'.join([
            self.template.get('input_beginning'),
            '\n'.join(elements_item_format),
        ])

    def __assmble_shots(
        self,
        shots_templates: List[object],
        example_num: int,
    ):
        selected_examples = np.random.choice(shots_templates, example_num, replace=False)
        res = []

        for idx, item in enumerate(selected_examples):
            elements_format = self.__assmble_elemets(item.get('elements'))
            generation_format = ('\n').join([
                self.template.get('output_beginning'),
                item.get('caption')
            ])
            res.append(elements_format + '\n' + generation_format)
        
        return ('\n').join(res)

    def __assemble_shot_template(
        self,
        template_type: str,
        example_num: int,
    ):
        # Starting
        part1 = ' '.join([
            self.template.get('start'),
            self.template.get('special_demands')[template_type],
            ' '.join(self.template.get('general_demands')),
        ])

        # Shots
        part2 = self.__assmble_shots(
            self.template.get('shots')[template_type],
            example_num,
        )

        # Assembling
        return '\n'.join([
            part1,
            part2,
        ])
    
    def generate_template(
        self,
        input_elements: List[str],
        template_type: str = 'short',
        example_num: int = 1
    ):
        shots_template = self.__assemble_shot_template(template_type, example_num)
        input_template = '\n'.join([
            self.__assmble_elemets(input_elements)
        ])

        return '\n'.join([
            shots_template,
            input_template,
            self.template.get('output_beginning')
        ])

class PromptGenerator:
    def __init__(self, model_name: str):
        self.model = GPT4All(
            model_name=model_name,
            allow_download=False,
            device='cuda'
        )

        self.prompt_template = PromptTemplate()

    def __post_process_prompts(self, output: str):
        # Only select the first none empty line as result
        for item in output.split('\n'):
            if not (item == ' ' or item  == ''):
                return item

    def generate_prompt(
        self,
        elements: List[str] = None,
        template_type: str = 'short',
        example_num: int = 1
    ):
        if elements is None:
            raise ValueError('Please input elements for prompt generation')
        
        prompt = self.prompt_template.generate_template(
            elements,
            template_type,
            example_num,
        )
        
        res = self.model.generate(
            prompt,
            max_tokens=100,
        )

        return self.__post_process_prompts(res)