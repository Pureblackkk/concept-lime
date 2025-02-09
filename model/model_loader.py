import torch

class ModelLoader:
    def __init__(self, device: torch.device):
        self.device = device

    def get_biased_model(self, path: str = None):
        '''
            Load biased trained model from given path
        '''
        if path is None:
            raise ValueError('Please input the biased model path')
        
        # Load biased model
        model = torch.load(path, map_location=self.device)
        
        return model