import json
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def remove_old_checkpoints(keep_last=2):
    path = './'
    configs_path = path + 'configs/mvlrs_v1/finished/'
    checkpoints_path = path + 'checkpoints/'
    for config_path in os.listdir(configs_path):
        config = json.load(open(configs_path + config_path, 'r'))
        checkpoint_path = checkpoints_path + config['experiment_path'] + '/' + config['experiment_name']
        checkpoint_files = os.listdir(checkpoint_path)
        checkpoint_files.sort(key=natural_keys)
        print(config['experiment_name'])
        print(len(checkpoint_files))
        print(checkpoint_files)
        if len(checkpoint_files) > 1 + 3 * keep_last:
            checkpoint_files = checkpoint_files[1:-3 * keep_last]
            print(checkpoint_files)
            for file in checkpoint_files:
                os.remove(checkpoint_path + '/' + file)

