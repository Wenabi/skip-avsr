import numpy as np
import json

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
              'encoder_units_per_layer': [(256,), (256,256,256)],
              'decoder_units_per_layer': (256,),
              'batch_size': (48, 48),
              'iterations': (500, 100),
              'learning_rate': (0.001, 0.0001),
              'max_label_length': 100,  # LRS3 150, LRS2 100
              'write_summary': False,
              'set_data_null':''}
    config.update(new_config)
    experiment_path = config['dataset']+'/'
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
    makedirs(path.dirname('./configs/'+config['dataset']+'/gpu_'+str(gpu_num)+'/'), exist_ok=True)
    with open('./configs/'+config['dataset']+'/gpu_'+str(gpu_num)+'/'+config['experiment_name']+'.json', 'w') as f:
        json.dump(config, f)
    createFolders(experiment_path)
    
def createConfigs(num_gpus):
    config_list = []
    for seed in range(3):
        dataset = 'mvlrs_v1'
        for architecture in ['bimodal']:
            for cell_type in [['skip_lstm', 'lstm', 'lstm'], ['lstm', 'skip_lstm', 'lstm'], ['lstm','lstm','skip_lstm']]:
                for cps in [0.0, 0.01, 0.001, 0.0001, 0.00001]:
                    config = {'seed':seed,
                              'dataset':dataset,
                              'architecture':architecture,
                              'cell_type':cell_type,
                              'cost_per_sample':[cps if ct == 'skip_lstm' else 0.0 for ct in cell_type],
                              'set_data_null':''}
                    config_list.append(config)
    print('Number of Configs:', len(config_list))
    for i in range(len(config_list)):
        x = i%num_gpus
        createConfig(x, config_list[i])
    makedirs(path.dirname('./configs/' + config['dataset'] + '/finished/'), exist_ok=True)
        
createConfigs(8)
