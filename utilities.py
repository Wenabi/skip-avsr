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

def remove_old_experiment_results():
    configs_path = 'N:/experiments/fourth experiments backup/configs/mvlrs_v1/all configs/'
    working_dir = 'F:/Documents/PycharmProjects/Masterthesis/skip-avsr/'
    for config_name in os.listdir(configs_path):
        config = json.load(open(configs_path + config_name, 'r'))
        try:
            for folder in ['logs','checkpoints','predictions']:
                path = working_dir + folder + '/' + config['experiment_path']
                if 'logs' == folder:
                    path = path + '/'
                else:
                    path = path + config['experiment_name'] + '/'
                for p in os.listdir(path):
                    if os.path.isdir(path+p):
                        for f in os.listdir(path+p):
                            os.remove(path+p+'/'+f)
                    else:
                        os.remove(path+p)
        except:
            print(config['experiment_name'])
        

remove_old_experiment_results()