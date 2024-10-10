import imageio
import torch
import numpy as np
import os, sys
from PIL import Image


base_dir = "results/dump/eqa_answer_given/episodes/"
save_dir = "movies/eqa_answer_given"
thread_list = os.listdir(base_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for thread in thread_list:
    thread_path = os.path.join(base_dir, thread)
    eps_list = os.listdir(thread_path)
    for eps in eps_list:
        path = os.path.join(base_dir, thread, eps)
        file_names = os.listdir(path)
        imgs = []
        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            image = Image.open(file_path)
            imgs.append(image)
        imageio.mimsave(f'{save_dir}/thread_{thread}_{eps}.mp4', imgs, fps=4)
