import os
import json
from pathlib import Path
import torch
import yaml
from tqdm import tqdm
import torch.optim as optim
from utils import load_model, transform
from get_loader import get_loader
from nlgmetricverse import NLGMetricverse, load_metric
from train import precompute_images, get_model, parse_args
from PIL import Image  # Load img

model_arch = 'stacked'

PREFIX = "/storage/ice1/0/7/agupta965/mscoco/val2014/"
fname = "000000002153.jpg"

img = Image.open(os.path.join(PREFIX, fname))
# Ensure the image has 3 channels
if img.mode != 'RGB':
    img = img.convert('RGB')

# Apply image transformations if provided
if self.transform is not None:
    img = self.transform(img)
