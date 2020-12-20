import numpy as np
import json
from pprint import pprint

from os import makedirs, path, system

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def createFolders(experiment_path):
    for folder in ['logs', 'predictions','checkpoints']:
        makedirs(path.dirname(path.join(folder,experiment_path)), exist_ok=True)

def createConfig(gpu_num, new_config):
    config = {'seed':0,
              'dataset': 'mvlrs_v1',
              'architecture': 'av_align',
              'cell_type': ['lstm','lstm','lstm'],
              'cost_per_sample': [0.0, 0.0, 0.0],
              'snr': 'clean',
              'encoder_units_per_layer': [(256,), (256,256,256)],
              'decoder_units_per_layer': (256,),
              'batch_size': (48, 48),
              'iterations': (100, 25),
              'learning_rate': (0.001, 0.0001),
              'max_label_length': 100,  # LRS3 150, LRS2 100
              'write_summary': False,
              'set_data_null':''}
    config.update(new_config)
    experiment_path = config['dataset']+'/'
    experiment_path += config['snr']+'/'
    experiment_path += config['architecture']+'/'
    skip_layer = np.array(['v','a','d'])[np.where(np.array(config['cell_type'])=='skip_lstm')]
    if len(skip_layer) == 0:
        skip_layer = ['n']
    experiment_path += ''.join(skip_layer)+'/'
    for cps in config['cost_per_sample']:
        experiment_path += format_e(cps) + '/'
    experiment_path += str(len(config['encoder_units_per_layer'][0]))+str(len(config['encoder_units_per_layer'][1]))+\
                       str(len(config['decoder_units_per_layer']))+'/'
    experiment_path += 'exp'+str(config['seed'])+'/'
    config['experiment_path'] = experiment_path
    config['experiment_name'] = experiment_path.replace('/','_')[:-1]
    makedirs(path.dirname('./configs/gpu_'+str(gpu_num)+'/'), exist_ok=True)
    with open('./configs/gpu_'+str(gpu_num)+'/'+config['experiment_name']+'.json', 'w') as f:
        json.dump(config, f)
    createFolders(experiment_path)
    
def createConfigs(gpus):
    config_list = []
    for seed in range(3):
        architecture, cell_type = 'av_align', ['skip_lstm','skip_lstm', 'skip_lstm']
        for snr in ['clean']:
            for cps_values in [[0.0001, 0.0001, 0.001],
                               [0.0001, 0.0005, 0.0001],
                               [0.00001, 0.0005, 0.0001]]:
                config = {'seed': seed,
                          'dataset': 'LRS3',
                          'snr': snr,
                          'architecture': architecture,
                          'cell_type': cell_type,
                          'cost_per_sample': cps_values,
                          'set_data_null': '',
                          'max_label_length': 150} #LRS3 150, mvlrs_v1 100
                config_list.append(config)
        for architecture in ['av_align', 'bimodal']:
            config = {'seed': seed,
                      'dataset': 'mvlrs_v1',
                      'snr': 'clean',
                      'architecture': architecture,
                      'cell_type': ['lstm','lstm', 'lstm'],
                      'cost_per_sample': [0.0, 0.0, 0.0],
                      'set_data_null': '',
                      'max_label_length': 100}  # LRS3 150, mvlrs_v1 100
            config_list.append(config)
        config = {'seed': seed,
                  'dataset': 'LRS3',
                  'snr': 'clean',
                  'architecture': 'av_align',
                  'cell_type': ['lstm', 'lstm', 'lstm'],
                  'cost_per_sample': [0.0, 0.0, 0.0],
                  'set_data_null': '',
                  'max_label_length': 150}  # LRS3 150, mvlrs_v1 100
        config_list.append(config)
            
    print('Number of Configs:', len(config_list))
    #pprint(config_list)
    for i in range(len(config_list)):
        x = i%len(gpus)
        createConfig(gpus[x], config_list[i])
    makedirs(path.dirname('./configs/finished/'), exist_ok=True)
    
def createConfigsTest(gpus):
    config_list = []
    dataset = 'LRS3'
    architecture = 'av_align'
    cell_type = ['skip_lstm','skip_lstm','skip_lstm']
    config = {'seed': 0,
              'dataset': dataset,
              'snr': 'clean',
              'architecture': architecture,
              'cell_type': cell_type,
              'cost_per_sample': [0.0001, 0.0001, 0.0001],
              'set_data_null': '',
              'batch_size': (48, 48)}
    config_list.append(config)
    for i in range(len(config_list)):
        x = i%len(gpus)
        createConfig(gpus[x], config_list[i])
    makedirs(path.dirname('./configs/finished/'), exist_ok=True)
    
createConfigs([0])
#createConfigsTest([0])