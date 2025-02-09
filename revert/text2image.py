from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from accelerate import PartialState
from typing import List
import json
import torch
import argparse
import os

DEFAULT_MODEL_PATH = '/home/pureblackkkk/my_volume/models/stable-diffusion-3.5-medium'

def run(
    extend_mode: bool,
    model_name_or_path: str = DEFAULT_MODEL_PATH,
    prompt_pair: List[tuple] = [],
):
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        text_encoder_3=None,
    )

    # Start distributed running
    distributed_state = PartialState()
    pipeline.enable_model_cpu_offload(int(str(distributed_state.device).split(':')[-1]))

    with distributed_state.split_between_processes(prompt_pair) as pairs:
        for prompt, save_dir in pairs:
            # If file exists and use extend_mode
            if os.path.exists(save_dir) and extend_mode:
                continue

            image = pipeline(
                prompt=prompt,
                num_inference_steps=40,
                guidance_scale=7,
                max_sequence_length=256,
            ).images[0]

            image.save(save_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    # Model name or path
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH)
    
    # Temp json file path
    parser.add_argument("--temp_json_path", type=str)

    # Is extend mode
    parser.add_argument("--extend_mode", action='store_true')

    args, _ = parser.parse_known_args()
    return args

def load_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        return [tuple(prompt_pair) for prompt_pair in data['prompt_pair']]


if __name__ == '__main__':
    # Load param
    args = parse_args()

    # Load json file pair
    prompt_pair = load_json_file(args.temp_json_path)

    # Run the image generation
    run(args.extend_mode, args.model_name_or_path, prompt_pair)


