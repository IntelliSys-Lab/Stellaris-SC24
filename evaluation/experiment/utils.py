import argparse
import torch
import numpy as np
import csv
import os
import config


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_outliers(x, m=2):
    data = np.array(x)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return data[s<m].tolist()

def scale(x, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((x - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]

def cv(x):
    return abs(np.std(x) / np.mean(x))

def z_score(l, x):
    return (x - np.mean(l)) / np.std(l)

def fuse(grad_list):
    return torch.concat([torch.reshape(grad, (-1,)) for grad in grad_list], -1)

def process_logp_ratio(logp_ratio):
    processed_logp_ratio = []
    for ratio in logp_ratio.tolist():
        ratio_abs = np.abs(ratio - 1)
        processed_logp_ratio.append(ratio_abs)
    # processed_logp_ratio = remove_outliers(processed_logp_ratio)
    
    logp_ratio_min = np.min(processed_logp_ratio)
    logp_ratio_mean = np.mean(processed_logp_ratio)
    logp_ratio_max = np.max(processed_logp_ratio)

    return logp_ratio_min, logp_ratio_mean, logp_ratio_max

#
# CSV
# 

def export_csv(
    scheduler_name,
    env_name, 
    algo_name, 
    csv_name,
    csv_file
):
    mkdir(config.log_path)
    with open(
        f"{config.log_path}/{scheduler_name}~{env_name}~{algo_name}~{csv_name}.csv",
        "w", 
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerows(csv_file)

# File management
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Parse arguments
def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default="Hopper-v3", help='Environment name')
    parser.add_argument('--algo_name', type=str, default="ppo", help='Algorithm name')
    args = parser.parse_args()

    return args