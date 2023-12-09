import os
import time
import random
import numpy as np
import cv2
import torch
import gdown



""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)




""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



""" Download model from gdrive """
def get_model_from_gdrive():
    file_id = "1OoXv3D8HN0Cm9Gl_IR0I4EAHMUUebPFQ"
    output = 'files/checkpoint.pth'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)