import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config

from torch.utils.data import Dataset

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    return parser


import os
import json
from PIL import Image
from torch.utils.data import Dataset

class VideoCrafterDataset(Dataset):
    def __init__(self, dir, mode="train", transform=None):
        self.dir = dir
        self.mode = mode
        self.transform = transform
        self.data = []
        
        img_paths = []
        prompts_dict = {}
        for relations in os.listdir(self.dir):    
            annotation_file = ""
            for images in os.listdir(os.path.join(self.dir, relations)):
                img_path = os.path.join(self.dir, relations, images)
                if img_path.endswith(".json"):
                    annotation_file = os.path.join(self.dir, relations, images)
                else:
                    img_paths.append(img_path)
            
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
            
            for image_path, prompts in annotations.items():
                image_path = os.path.join(self.dir, relations, image_path)
                prompts_dict[image_path] = prompts

        for img_path in img_paths:
            for i in range(len(prompts_dict[img_path])):
                prompt = prompts_dict[img_path][i]
                prompt = prompt.replace("{}", "<R>")
                self.data.append((img_path, prompt))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, prompt = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        return image, prompt

if __name__ == '__main__':
    dir = "/home/jvthunder/URECA/reversion_benchmark_v1"
    dataset = VideoCrafterDataset(dir)
    print(len(dataset))

    for i in range(len(dataset)):
        image, prompt = dataset[i]
        print(image, prompt)
        
    

# def run_inference(args, gpu_num, gpu_no, **kwargs):
#     ## step 1: model config
#     ## -----------------------------------------------------------------
#     config = OmegaConf.load(args.config)
#     #data_config = config.pop("data", OmegaConf.create())
#     model_config = config.pop("model", OmegaConf.create())
#     model = instantiate_from_config(model_config)
#     model = model.cuda(gpu_no)
#     assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
#     model = load_model_checkpoint(model, args.ckpt_path)
#     model.eval()

#     ## sample shape
#     assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
#     ## latent noise shape
#     h, w = args.height // 8, args.width // 8
#     frames = model.temporal_length if args.frames < 0 else args.frames
#     channels = model.channels


    

# if __name__ == '__main__':
#     print("@CoLVDM Training")
#     parser = get_parser()
#     args = parser.parse_args()
#     seed_everything(args.seed)
#     gpu_num, rank = 0, 0
#     run_inference(args, gpu_num, rank)