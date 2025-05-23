import argparse
import subprocess
import time

batch_size = [32, 64, 128]
learning_rate = [0.0005, 0.001, 0.005, 0.01]
num_layers = [1, 2, 4]
embed_size = [256, 512, 1024]
# config_file = ["config-flickr-cnnrnn.yaml", "config-flickr-cnnattn.yaml", "config-flickr-vitcnnattn.yaml"]
config_file = ["stack-flickr.yaml"]

for config in config_file:
    if "attn" in config:
        for bs in batch_size:
            for lr in learning_rate:
                for es in embed_size:
                    for nl in num_layers:
                        cmd = f"python train.py --config_file={config} --batch_size={bs} --learning_rate={lr} --num_layers={nl} --embed_size={es}"
                        subprocess.call(cmd, shell=True)
    else:
        for bs in batch_size:
            for lr in learning_rate:
                for es in embed_size:
                    cmd = f"python train.py --config_file={config} --batch_size={bs} --learning_rate={lr} --embed_size={es}"
                    subprocess.call(cmd, shell=True)
