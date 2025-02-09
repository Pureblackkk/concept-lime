from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import Any
import numpy as np
import os
from PIL import Image
import torch

class GradCAMER:
    def __init__(
        self,
        model: Any,
        revert_img_dir: str,
    ):
        self.model = model
        self.target_layer = self.__get_target_layer(model)
        
        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layer,
        )

        self.save_dir = self.__get_new_img_dir(revert_img_dir)

        # Create save dir if not exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
    
    def __get_new_img_dir(self, src_dir):
        # Get root dir of src dir
        parent_dir, current_folder = os.path.split(src_dir)

        return os.path.join(parent_dir, f'gradacm_{str(current_folder)}')
    
    def __save_cam_img(self, img, revert_img_name, pred_class):
        save_file = os.path.join(
            self.save_dir,
            f'{pred_class}#{revert_img_name}'
        )

        # Convert to
        image = Image.fromarray(img)
        image.save(f'{str(save_file)}')
        
    def __get_target_layer(self, model: Any):
        match model.__class__.__name__:
            case 'ResNet':
                return [model.layer4[-1]]
            case _:
                raise ValueError('Model target layer not implemented')
    
    def generate_cam(
        self,
        input_tensor,
        rgb_img,
        img_name,
        idx,
        outputs,
    ):
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=None,
        )
        
        rbg_img_float = np.float32(rgb_img) / 255
        pred_class = torch.argmax(outputs, dim=-1)

        for i in range(len(outputs)):
            vis_img = show_cam_on_image(rbg_img_float[i, :], grayscale_cam[i, :], use_rgb=True)
            self.__save_cam_img(
                vis_img,
                img_name[i],
                pred_class[i]
            )






        