import json
import os
import re
import pickle as p
import pandas as pd
import numpy as np
from pprint import pprint
from scipy.stats import pearsonr

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def remove_old_checkpoints(keep_last=2):
    for checkpoint_path, dir, checkpoint_files in os.walk('./checkpoints/'):
        if len(checkpoint_files) > 0 and len(dir) == 0:
            checkpoint_files.sort(key=natural_keys)
            print(len(checkpoint_files), checkpoint_path)
            #print(checkpoint_files)
            if len(checkpoint_files) > 1 + 3 * keep_last:
                checkpoint_files = checkpoint_files[1:-3 * keep_last]
                #print(checkpoint_files)
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

'''Returns the update, error, train_error rates.'''
def getExperimentResults(log_path, two_d_cps=False):
    error_rate, train_error_rate, v_update_rate, a_update_rate = 0.0, 0.0, 0.0, 0.0
    if two_d_cps: dv_update_rate, da_update_rate = 0.0, 0.0
    else: d_update_rate = 0.0
    try:
        f = open(log_path, 'r')
    except:
        old_files = os.listdir(os.path.dirname(log_path))
        if len(old_files) == 1:
            old_file = old_files[0]
            os.rename(os.path.dirname(log_path)+'/'+old_file, log_path)
        f = open(log_path, 'r')
    lines = f.read().splitlines()
    for line in lines:
        if 'c_error_rate' in line:
            er = float(line.split(' ')[1][:-1])
            if 'train' in line: train_error_rate = er
            else: error_rate = er
        elif 'Updated_States_Rate' in line:
            er = float(line.split(' ')[0].split(':')[1])*100
            if 'Decoder' in line:
                if not two_d_cps: d_update_rate = er
                elif 'Video' in line: dv_update_rate = er
                elif 'Audio' in line: da_update_rate = er
            elif 'Video' in line: v_update_rate = er
            elif 'Audio' in line: a_update_rate = er
    if two_d_cps:
        return error_rate, train_error_rate, v_update_rate, a_update_rate, dv_update_rate, da_update_rate
    else:
        return error_rate, train_error_rate, v_update_rate, a_update_rate, d_update_rate
        
def getResults():
    try:
        data = p.load(open('./logs/results.p', 'rb'))
    except:
        data = {}
    return data
    
def storeExperimentResults():
    data = getResults()
    path = './configs/mvlrs_v1/finished/'
    for config_name in os.listdir(path):
        config = json.load(open(path+config_name, 'r'))
        print(config['experiment_path'])
        if not config['experiment_name'] in data.keys():
            log_path = './logs/'+config['experiment_path']+config['experiment_name']
            if len(config['cost_per_sample']) == 3:
                error_rate, train_error_rate, v_update_rate, a_update_rate, d_update_rate = getExperimentResults(log_path, False)
                data[config['experiment_path']] = {'Error Rate':error_rate, 'Train Error Rate':train_error_rate}
                if v_update_rate != 0.0: data[config['experiment_path']]['Video UR'] = v_update_rate
                if a_update_rate != 0.0: data[config['experiment_path']]['Audio UR'] = a_update_rate
                if d_update_rate != 0.0: data[config['experiment_path']]['Decoder UR'] = d_update_rate
            else:
                error_rate, train_error_rate, v_update_rate, a_update_rate, dv_update_rate, da_update_rate = getExperimentResults(log_path, True)
                data[config['experiment_path']] = {'Error Rate': error_rate, 'Train Error Rate': train_error_rate}
                if v_update_rate != 0.0: data[config['experiment_path']]['Video UR'] = v_update_rate
                if a_update_rate != 0.0: data[config['experiment_path']]['Audio UR'] = a_update_rate
                if dv_update_rate != 0.0: data[config['experiment_path']]['Decoder Video UR'] = dv_update_rate
                if da_update_rate != 0.0: data[config['experiment_path']]['Decoder Audio UR'] = da_update_rate
    p.dump(data, open('./logs/results.p', 'wb'))
    
def getHyperParams(experiment_path):
    params = experiment_path.split('/')[:-1]
    return params
    
'''Deals with combining two values into a single list.'''
def mergeTwoValues(a, b):
    if type(a) == list:
        a.append(b)
        return a
    else:
        return [a, b]

'''Two dict with same key get merged, so that the values of each key are now lists of the values of each dict'''
def mergeData(dict_a, dict_b):
    return {key:mergeTwoValues(dict_a[key], dict_b[key]) for key in dict_a.keys()}
    
def formatCPS(cps):
    return format(eval(cps), 'f').rstrip('0').rstrip('.')
    
def organizedResults():
    data = getResults()
    dataframe_data = {}
    for experiment_path in data.keys():
        values = []
        if 'bimodal' in experiment_path:
            dataset, snr, arch, skipped_layers, v_cps, a_cps, dv_cps, da_cps, layers, exp_nr = getHyperParams(experiment_path)
            cps = []
            if 'v' in skipped_layers: cps.append(formatCPS(v_cps))
            if 'a' in skipped_layers: cps.append(formatCPS(a_cps))
            if 'd' in skipped_layers: cps.append(formatCPS(dv_cps)); cps.append(formatCPS(da_cps))
            for metric in data[experiment_path].keys():
                values.append(['_'.join(cps), data[experiment_path][metric], metric])
        else:
            print(experiment_path)
            dataset, snr, arch, skipped_layers, v_cps, a_cps, d_cps, layers, exp_nr = getHyperParams(experiment_path)
            cps = []
            if 'v' in skipped_layers: cps.append(formatCPS(v_cps))
            if 'a' in skipped_layers: cps.append(formatCPS(a_cps))
            if 'd' in skipped_layers: cps.append(formatCPS(d_cps))
            for metric in data[experiment_path].keys():
                values.append(['_'.join(cps), data[experiment_path][metric], metric])
        key = snr+'_'+arch + '_' + skipped_layers
        if key in dataframe_data.keys():
            dataframe_data[key] = dataframe_data[key] + values
        else:
            dataframe_data[key] = values
    return dataframe_data

def resultsToCSV():
    dataframe_data = organizedResults()
    dataframes_single = {}
    metricOrder = ['Error Rate', 'Train Error Rate', 'Video UR', 'Audio UR', 'Decoder UR', 'Decoder Video UR', 'Decoder Audio UR']
    for key in [key for key in dataframe_data.keys()]:
        print(key)
        arch = '_'.join(key.split('_')[:-1])
        skipped_layers = key.split('_')[-1]
        if arch == 'av_align':
            cps_index = ['CPS V', 'CPS A', 'CPS D']
            cps_index = [cps_index['vad'.index(c)] for c in skipped_layers]
        else:
            cps_index = ['CPS V', 'CPS A', 'CPS D A']
            cps_index = [cps_index['vad'.index(c)] for c in skipped_layers]
            if 'd' in skipped_layers:
                cps_index.append('CPS D V')
        df = pd.DataFrame(dataframe_data[key], columns=['CostPerSample', 'Rate', 'Metric'])
        
        df[cps_index] = df.CostPerSample.str.split('_', expand=True,)
        
        metrics = set([v[2] for v in df.values])
        column_order = [metricOrder[i] for i in sorted([metricOrder.index(m) for m in metrics])]
        
        df_pivot = df.pivot_table(index=cps_index, columns='Metric', values='Rate', aggfunc=np.mean)
        #print(df_pivot['Error Rate'].index.values)
        #a = [float(x) for x in df_pivot['Error Rate'].index.values]
        #b = df_pivot['Error Rate'].values
        #print(a)
        #print(b)
        #print(pearsonr(a, b))
        csv = df_pivot.reindex(column_order, axis=1).to_csv()
        print(csv)
        
#storeExperimentResults()
#resultsToCSV()

def add_parameter_to_config():
    configs_path = './configs/mvlrs_v1/finished/'
    for file in os.listdir(configs_path):
        config = json.load(open(configs_path + file, 'r'))
        if not 'clean' in file:
            config['snr'] = 'clean'
            old_experiment_path = config['experiment_path']
            if not 'clean' in old_experiment_path:
                experiment_path = old_experiment_path.replace('mvlrs_v1', 'mvlrs_v1/clean')
                config['experiment_path'] = experiment_path
            old_experiment_name = config['experiment_name']
            if not 'clean' in old_experiment_name:
                experiment_name = old_experiment_name.replace('mvlrs_v1', 'mvlrs_v1_clean')
                config['experiment_name'] = experiment_name
            json.dump(config, open(configs_path+file, 'w'))
            if not os.path.exists(configs_path+experiment_name+'.json'):
                os.rename(configs_path+file, configs_path+experiment_name+'.json')
        else:
            experiment_path = config['experiment_path']
            experiment_name = config['experiment_name']
            old_experiment_path = experiment_name.replace('/clean', '')
            old_experiment_name = experiment_name.replace('_clean', '')
        print('./checkpoints/'+experiment_path+experiment_name)
        if not os.path.exists('./checkpoints/' + experiment_path + experiment_name):
            os.rename('./checkpoints/' + experiment_path + old_experiment_name,
                      './checkpoints/' + experiment_path + experiment_name)
        if not os.path.exists('./predictions/'+experiment_path+experiment_name):
            os.rename('./predictions/' + experiment_path + old_experiment_name,
                      './predictions/' + experiment_path + experiment_name)
        if not os.path.exists('./logs/'+experiment_path+experiment_name):
            os.rename('./logs/' + experiment_path + old_experiment_name, './logs/' + experiment_path + experiment_name)
        
#add_parameter_to_config()

import shutil
def old_path_to_new_path():
    path = './checkpoints/mvlrs_v1/clean/av_align/v/'
    for folder in ['00', '001', '0001', '00001', '000001']:
        original = path+folder+'/131'
        t = "{:.0E}".format(float(folder[0]+'.'+folder[1:]))
        target = path+str(t)+'/0E+00'+'/0E+00'+'/'
        try:
            shutil.rmtree(target+'131')
        except:
            pass
        print(original)
        print(target)
        try:
            shutil.move(original, target)
        except:
            pass

        target = path + str(t)+'/0E+00'+'/0E+00' + '/131/'
        print(os.listdir(target))
        for exp_nr in os.listdir(target):
            for file in os.listdir(target + exp_nr):
                if file.startswith('mvlrs'):
                    print(file)
                    os.rename(target + exp_nr + '/' + file,
                              target + exp_nr + '/' + file.replace(folder, str(t)+'_0E+00'+'_0E+00'))

#old_path_to_new_path()
def test():
    for root, dirs, files in os.walk('./checkpoints/mvlrs_v1/clean/bimodal/va/'):
        if len(dirs) > 0 and not '5E-04' in root:
            for dir in dirs:
                if dir.startswith('mvlrs'):
                    print(root, dir)
                    print(root+'/'+dir.replace(dir[-8:], dir[-14:]))
                    os.rename(root+'/'+dir, root+'/'+dir.replace(dir[-8:], dir[-14:]))
        
#test()

#storeExperimentResults()

def exportAudioUpdateStates():
    configs_path = './configs/mvlrs_v1/noise_configs/'
    for config_file in os.listdir(configs_path):
        config = json.load(open(configs_path+config_file, 'r'))
        experiment_path = config['experiment_path']
        experiment_name = config['experiment_name']
        audio_updates_states = {}
        for eval_file in os.listdir('./eval_data/'+experiment_path+experiment_name):
            eval_data = p.load(open('./eval_data/'+experiment_path+experiment_name+'/'+eval_file, 'rb'))
            for key, value in eval_data.items():
                audio_updates_states[key] = value['audio_updated_states']
        p.dump(audio_updates_states, open('./eval_data/'+experiment_path+experiment_name+'/audio_updated_states.p', 'wb'))
        
remove_old_checkpoints()