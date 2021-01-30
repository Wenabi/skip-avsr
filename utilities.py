import json
import os
import re
import pickle as p
import pandas as pd
import glob
pd.options.display.float_format = '{:,.2f}'.format
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
            # print(checkpoint_files)
            if len(checkpoint_files) > 1 + 3 * keep_last:
                checkpoint_files = checkpoint_files[1:-3 * keep_last]
                # print(checkpoint_files)
                for file in checkpoint_files:
                    os.remove(checkpoint_path + '/' + file)


def remove_old_experiment_results():
    configs_path = 'N:/experiments/fourth experiments backup/configs/mvlrs_v1/all configs/'
    working_dir = 'F:/Documents/PycharmProjects/Masterthesis/skip-avsr/'
    for config_name in os.listdir(configs_path):
        config = json.load(open(configs_path + config_name, 'r'))
        try:
            for folder in ['logs', 'checkpoints', 'predictions']:
                path = working_dir + folder + '/' + config['experiment_path']
                if 'logs' == folder:
                    path = path + '/'
                else:
                    path = path + config['experiment_name'] + '/'
                for p in os.listdir(path):
                    if os.path.isdir(path + p):
                        for f in os.listdir(path + p):
                            os.remove(path + p + '/' + f)
                    else:
                        os.remove(path + p)
        except:
            print(config['experiment_name'])


'''Returns the update, error, train_error rates.'''


def getExperimentResults(log_path, two_d_cps=False):
    c_error_rate, train_cer, v_update_rate, w_error_rate, train_wer, v_update_rate, a_update_rate = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if two_d_cps:
        dv_update_rate, da_update_rate = 0.0, 0.0
    else:
        d_update_rate = 0.0
    try:
        f = open(log_path, 'r')
    except:
        old_files = os.listdir(os.path.dirname(log_path))
        if len(old_files) == 1:
            old_file = old_files[0]
            os.rename(os.path.dirname(log_path) + '/' + old_file, log_path)
        f = open(log_path, 'r')
    lines = f.read().splitlines()
    for line in lines:
        if 'error_rate' in line:
            cer = float(line.split(' ')[1][:-1])
            wer = float(line.split(' ')[9][:-1])
            if 'train' in line:
                train_cer = cer
                train_wer = wer
            else:
                c_error_rate = cer
                w_error_rate = wer
        
        elif 'Updated_States_Rate' in line:
            er = float(line.split(' ')[0].split(':')[1]) * 100
            if 'Decoder' in line:
                if not two_d_cps:
                    d_update_rate = er
                elif 'Video' in line:
                    dv_update_rate = er
                elif 'Audio' in line:
                    da_update_rate = er
            elif 'Video' in line:
                v_update_rate = er
            elif 'Audio' in line:
                a_update_rate = er
    if two_d_cps:
        return c_error_rate, train_cer, w_error_rate, train_wer, v_update_rate, a_update_rate, dv_update_rate, da_update_rate
    else:
        return c_error_rate, train_cer, w_error_rate, train_wer, v_update_rate, a_update_rate, d_update_rate


def getResults():
    try:
        data = p.load(open('./logs/results.p', 'rb'))
    except:
        data = {}
    return data


def storeExperimentResults():
    data = getResults()
    path = './configs/finished/'
    for config_name in os.listdir(path):
        config = json.load(open(path + config_name, 'r'))
        print(config['experiment_path'])
        if not config['experiment_name'] in data.keys():
            log_path = './logs/' + config['experiment_path'] + config['experiment_name']
            if len(config['cost_per_sample']) == 3:
                c_error_rate, c_train_error_rate, w_error_rate, w_train_error_rate, v_update_rate, a_update_rate, d_update_rate = getExperimentResults(
                    log_path, False)
                data[config['experiment_path']] = {'CER': c_error_rate, 'Train CER': c_train_error_rate,
                                                   'WER': w_error_rate, 'Train WER': w_train_error_rate}
                if v_update_rate != 0.0: data[config['experiment_path']]['VUR'] = v_update_rate
                if a_update_rate != 0.0: data[config['experiment_path']]['AUR'] = a_update_rate
                if d_update_rate != 0.0: data[config['experiment_path']]['DUR'] = d_update_rate
            else:
                c_error_rate, c_train_error_rate, w_error_rate, w_train_error_rate, v_update_rate, a_update_rate, dv_update_rate, da_update_rate = getExperimentResults(
                    log_path, True)
                data[config['experiment_path']] = {'CER': c_error_rate, 'Train CER': c_train_error_rate,
                                                   'WER': w_error_rate, 'Train WER': w_train_error_rate}
                if v_update_rate != 0.0: data[config['experiment_path']]['VUR'] = v_update_rate
                if a_update_rate != 0.0: data[config['experiment_path']]['AUR'] = a_update_rate
                if dv_update_rate != 0.0: data[config['experiment_path']]['D VUR'] = dv_update_rate
                if da_update_rate != 0.0: data[config['experiment_path']]['D AUR'] = da_update_rate
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
    return {key: mergeTwoValues(dict_a[key], dict_b[key]) for key in dict_a.keys()}


def formatCPS(cps):
    return format(eval(cps), 'f').rstrip('0').rstrip('.')


def organizedResults(getExp=False):
    data = getResults()
    dataframe_data = {}
    for experiment_path in data.keys():
        values = []
        # print(experiment_path)
        if 'bimodal' in experiment_path:
            dataset, snr, arch, skipped_layers, v_cps, a_cps, dv_cps, da_cps, layers, exp_nr = getHyperParams(
                experiment_path)
            cps = []
            if 'v' in skipped_layers: cps.append(v_cps)
            if 'a' in skipped_layers: cps.append(a_cps)
            if 'd' in skipped_layers: cps.append(dv_cps); cps.append(da_cps)
            for metric in data[experiment_path].keys():
                values.append([' '.join(cps), round(data[experiment_path][metric],2), metric])
        else:
            dataset, snr, arch, skipped_layers, v_cps, a_cps, d_cps, layers, exp_nr = getHyperParams(experiment_path)
            cps = []
            if 'v' in skipped_layers: cps.append(v_cps)
            if 'a' in skipped_layers: cps.append(a_cps)
            if 'd' in skipped_layers: cps.append(d_cps)
            if getExp:
                for metric in data[experiment_path].keys():
                    values.append([' '.join(cps), round(data[experiment_path][metric],2), metric, exp_nr])
            else:
                for metric in data[experiment_path].keys():
                    values.append([' '.join(cps), round(data[experiment_path][metric],2), metric])
        key = dataset + '---' + snr + '---' + arch + '---' + skipped_layers
        if key in dataframe_data.keys():
            dataframe_data[key] = dataframe_data[key] + values
        else:
            dataframe_data[key] = values
    return dataframe_data


def resultsToCSV():
    remove_metric = ['Train CER', 'Train WER']
    dataframe_data = organizedResults()
    metricOrder = ['CER', 'Train CER', 'WER', 'Train WER', 'VUR', 'AUR', 'DUR', 'D VUR', 'D AUR']
    for key in [key for key in dataframe_data.keys()]:
        print(key)
        arch = key.split('---')[2]
        skipped_layers = key.split('---')[-1]
        if arch == 'av_align':
            cps_index = ['', 'CPS A', 'CPS D', 'CPS V']
            cps_index = [cps_index['nvad'.index(c)] for c in skipped_layers]
        else:
            cps_index = ['', 'CPS V', 'CPS A', 'CPS D A']
            cps_index = [cps_index['nvad'.index(c)] for c in skipped_layers]
            if 'd' in skipped_layers:
                cps_index.append('CPS D V')
        df = pd.DataFrame(dataframe_data[key], columns=['CostPerSample', 'Rate', 'Metric'])
        
        # if not 'n' in skipped_layers and key.split('---')[0] == 'mvlrs_v1' and 'clean' in key:
        #    single_data = {}
        #    for skipped_layer in 'nvad':
        #        single = dataframe_data['---'.join(key.split('---')[:-1])+'---'+skipped_layer]
        #        single = pd.DataFrame(single, columns=['CPS', 'Rate', 'Metric']).groupby('CPS').mean()
        #        print(single)
        #        single_data[skipped_layer] = single
        df[cps_index] = df.CostPerSample.str.split(' ', expand=True, )
        
        metrics = set([v[2] for v in df.values if v[2] not in remove_metric])
        column_order = [metricOrder[i] for i in sorted([metricOrder.index(m) for m in metrics])]
        
        df_pivot = df.pivot_table(index=cps_index, columns='Metric', values='Rate', aggfunc=np.mean)
        # if not 'n' in skipped_layers and key.split('---')[0] == 'mvlrs_v1' and 'clean' in key:
        #    new_columns = []
        #    for col in column_order:
        #        if 'UR' in col and  len(key.split('---')[-1]) > 1 :
        #            df_pivot['Diff '+col] = df_pivot[col]-single_data[col[0].lower()]['Rate'][col]
        #        elif 'ER' in col:
        #            df_pivot['Diff ' + col] = df_pivot[col] - single_data['n']['Rate'][col]
        #        new_columns.append('Diff '+col)
        #    column_order = column_order + new_columns
        # print(df_pivot['Error Rate'].index.values)
        # a = [float(x) for x in df_pivot['Error Rate'].index.values]
        # b = df_pivot['Error Rate'].values
        # print(a)
        # print(b)
        # print(pearsonr(a, b))
        csv = df_pivot.reindex(column_order, axis=1).to_csv()
        print(csv)


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
            json.dump(config, open(configs_path + file, 'w'))
            if not os.path.exists(configs_path + experiment_name + '.json'):
                os.rename(configs_path + file, configs_path + experiment_name + '.json')
        else:
            experiment_path = config['experiment_path']
            experiment_name = config['experiment_name']
            old_experiment_path = experiment_name.replace('/clean', '')
            old_experiment_name = experiment_name.replace('_clean', '')
        print('./checkpoints/' + experiment_path + experiment_name)
        if not os.path.exists('./checkpoints/' + experiment_path + experiment_name):
            os.rename('./checkpoints/' + experiment_path + old_experiment_name,
                      './checkpoints/' + experiment_path + experiment_name)
        if not os.path.exists('./predictions/' + experiment_path + experiment_name):
            os.rename('./predictions/' + experiment_path + old_experiment_name,
                      './predictions/' + experiment_path + experiment_name)
        if not os.path.exists('./logs/' + experiment_path + experiment_name):
            os.rename('./logs/' + experiment_path + old_experiment_name, './logs/' + experiment_path + experiment_name)


# add_parameter_to_config()

import shutil


def old_path_to_new_path():
    path = './checkpoints/mvlrs_v1/clean/av_align/v/'
    for folder in ['00', '001', '0001', '00001', '000001']:
        original = path + folder + '/131'
        t = "{:.0E}".format(float(folder[0] + '.' + folder[1:]))
        target = path + str(t) + '/0E+00' + '/0E+00' + '/'
        try:
            shutil.rmtree(target + '131')
        except:
            pass
        print(original)
        print(target)
        try:
            shutil.move(original, target)
        except:
            pass
        
        target = path + str(t) + '/0E+00' + '/0E+00' + '/131/'
        print(os.listdir(target))
        for exp_nr in os.listdir(target):
            for file in os.listdir(target + exp_nr):
                if file.startswith('mvlrs'):
                    print(file)
                    os.rename(target + exp_nr + '/' + file,
                              target + exp_nr + '/' + file.replace(folder, str(t) + '_0E+00' + '_0E+00'))


# old_path_to_new_path()
def test():
    for root, dirs, files in os.walk('./checkpoints/mvlrs_v1/clean/bimodal/va/'):
        if len(dirs) > 0 and not '5E-04' in root:
            for dir in dirs:
                if dir.startswith('mvlrs'):
                    print(root, dir)
                    print(root + '/' + dir.replace(dir[-8:], dir[-14:]))
                    os.rename(root + '/' + dir, root + '/' + dir.replace(dir[-8:], dir[-14:]))


# test()

# storeExperimentResults()

def exportAudioUpdateStates():
    configs_path = './configs/mvlrs_v1/noise_configs/'
    for config_file in os.listdir(configs_path):
        config = json.load(open(configs_path + config_file, 'r'))
        experiment_path = config['experiment_path']
        experiment_name = config['experiment_name']
        audio_updates_states = {}
        for eval_file in os.listdir('./eval_data/' + experiment_path + experiment_name):
            eval_data = p.load(open('./eval_data/' + experiment_path + experiment_name + '/' + eval_file, 'rb'))
            for key, value in eval_data.items():
                audio_updates_states[key] = value['audio_updated_states']
        p.dump(audio_updates_states,
               open('./eval_data/' + experiment_path + experiment_name + '/audio_updated_states.p', 'wb'))


def create_label_length():
    label_lengths = {}
    for filepath, folder, files in os.walk('F:/Documents/datasets/mvlrs_v1/main/'):
        if folder == []:
            for file in files:
                if file.endswith('txt'):
                    with open(filepath + '/' + file, 'r') as f:
                        text = f.read().splitlines()[0].split('  ')[1]
                        length = len(text)
                        key = '/'.join(filepath.split('/')[-2:]) + '/' + file.split('.')[0]
                        label_lengths[key] = {'length': length, 'text': text}
    p.dump(label_lengths, open('./datasets/mvlrs_v1/label_length.p', 'wb'))


def getLastIteration():
    configs_path = './configs/finished/'
    last_iterations = {}
    for config_file in os.listdir(configs_path):
        config = json.load(open(configs_path + config_file, 'r'))
        experiment_path = config['experiment_path']
        experiment_name = config['experiment_name']
        files = sorted(os.listdir('./checkpoints/' + experiment_path + experiment_name))
        # print(experiment_name[:-4], files[-1].split('-')[-1].split('.')[0])
        last_iterations[experiment_name[:-5]] = int(files[-1].split('-')[-1].split('.')[0])
    for key, value in last_iterations.items():
        print(key, np.mean(value))

def combine_mean_std(x):
    mean = np.mean(x)
    std = np.std(x)
    if std < 5:
        ret = '{:,.2f}'.format(mean)
    else:
        ret = '{:,.2f}'.format(mean) + '$+/-$' + '{:,.2f}'.format(std)
    return ret

def csvToLatex(csv, key):
    latex_lines = '\\begin{table}\n\centering\n\\begin{adjustbox}{width=\\textwidth}\n\\begin{tabular}'
    lines = csv.split('\r\n')[:-1]
    row = 1
    latex_lines += '{' + '|c' + ''.join(['|c' for _ in lines[0].split(',')]) + '|' + '}\n\\hline\n'
    columns = []
    new_lines = [lines[0]]
    new_lines += reversed(lines[1:])
    lines = new_lines
    for line in lines:
        if line.startswith(','):
            line = line[1:]
        if 'CER' in line:
            columns = line.split(',')
            line = 'Row & ' + line
            line = line.replace('Diff ', '$\Delta$')
        else:
            # split_line = line.split(',')
            # print(split_line)
            # print(lines[0].split(',')[1:])
            # print(std_pivot.keys())
            # new_line = []
            # for j, key in enumerate(lines[0].split(',')[1:]):
            #    print(j, key)
            #    if not 'CPS' in key and not '' == key and not 'Diff' in key and not 'ER' in key:
            #        print(len(skipped_layers))
            #        print(split_line[0:len(skipped_layers)])
            #        print(std_pivot.index.values)
            #        if len(skipped_layers) == 1 or (arch == "bimodal" and 'd' not in skipped_layers):
            #            std = std_pivot[key][split_line[0]]
            #        else:
            #            if arch == 'bimodal' and 'd' in skipped_layers:
            #                print('bimodal_d', tuple(split_line[0:len(skipped_layers)+1]))
            #                std = std_pivot[key][tuple(split_line[0:len(skipped_layers)+1])]
            #            else:
            #                std = std_pivot[key][tuple(split_line[0:len(skipped_layers)])]
            #        print(split_line)
            #        print('std', std)
            #        new_line.append(str(split_line[j+1]) +'+/-'+ '{:,.2f}'.format(std))
            #    else:
            #        new_line.append(split_line[j+1])
            values = line.split(',')
            new_values = []
            for i, x in enumerate(values):
                if not 'CPS' in columns[i]:
                    new_values.append(str(round(float(x), 2)))
                else:
                    new_values.append(x)
            line = ','.join(new_values)
            line = str(row) + ' & ' + line
            row += 1
        line = line.replace(',', ' & ')
        line += ' \\\\\n'
        line += '\\hline '
        latex_lines += line
    info = key.replace('---', ' ').replace('_', '-').upper()
    latex_lines += '\\end{tabular}\\label{' + info + '}\n\\end{adjustbox}\n'
    latex_lines += '\\caption{' + info + '}\n\\end{table}'
    print(latex_lines)

def resultsToLatex(showDiff = True):
    remove_metric = ['Train CER', 'Train WER']
    dataframe_data = organizedResults()
    metricOrder = ['CER', 'Train CER', 'WER', 'Train WER', 'VUR', 'AUR', 'DUR', 'D VUR', 'D AUR']
    for key in [key for key in dataframe_data.keys()]:
        print()
        print(key)
        print()
        arch = key.split('---')[2]
        skipped_layers = key.split('---')[-1]
        if arch == 'av_align':
            cps_index = ['', 'CPS V', 'CPS A', 'CPS D']
            cps_index = [cps_index['nvad'.index(c)] for c in skipped_layers]
        else:
            cps_index = ['', 'CPS V', 'CPS A', 'CPS D A']
            cps_index = [cps_index['nvad'.index(c)] for c in skipped_layers]
            if 'd' in skipped_layers:
                cps_index.append('CPS D V')
        df = pd.DataFrame(dataframe_data[key], columns=['CPS', 'Rate', 'Metric'])
        
        if not 'n' in skipped_layers and key.split('---')[0] == 'mvlrs_v1' and 'clean' in key:
            single_data = {}
            for skipped_layer in 'nvad':
                single = dataframe_data['---'.join(key.split('---')[:-1]) + '---' + skipped_layer]
                single = pd.DataFrame(single, columns=['CPS', 'Rate', 'Metric']).groupby(['CPS', 'Metric']).mean()
                single_data[skipped_layer] = single
        df[cps_index] = df.CPS.str.split(' ', expand=True, )
        
        metrics = set([v[2] for v in df.values if v[2] not in remove_metric])
        column_order = [metricOrder[i] for i in sorted([metricOrder.index(m) for m in metrics])]

        std_pivot = df.pivot_table(index=cps_index, columns='Metric', values='Rate', aggfunc=np.mean)
        df_pivot = df.pivot_table(index=cps_index, columns='Metric', values='Rate', aggfunc=np.mean)
        
        if not 'n' in skipped_layers and key.split('---')[0] == 'mvlrs_v1' and 'clean' in key and showDiff:
            new_columns = []
            for col in column_order:
                if 'UR' in col and len(key.split('---')[-1]) > 1:
                    std_pivot['Diff ' + col] = ""
                    if arch == "bimodal" and len(col) > 3:
                        lf = 'CPS ' + col[0:3]
                    else:
                        lf = 'CPS ' + col[0]
                    for k in df[lf].unique():
                        if arch == "bimodal" and col[0] == "D":
                            std_pivot.loc[(df_pivot.index.get_level_values(lf) == k), 'Diff ' + col] = \
                            df_pivot.loc[(df_pivot.index.get_level_values(lf) == k)][col] - \
                            single_data[col[0].lower()]['Rate'][(k+' '+k, col)]
                        else:
                            std_pivot.loc[(df_pivot.index.get_level_values(lf) == k), 'Diff ' + col] = \
                                df_pivot.loc[(df_pivot.index.get_level_values(lf) == k)][col] - \
                                single_data[col[0].lower()]['Rate'][(k, col)]
                    new_columns.append('Diff ' + col)
                # print(single_data[col[0].lower()]['Rate'].keys())
                # print(single_data[col[0].lower()]['Rate'][col])
                # df_pivot['Diff '+col] = df_pivot[col]-single_data[col[0].lower()]['Rate'][col]
                elif 'ER' in col:
                    std_pivot['Diff ' + col] = df_pivot[col] - single_data['n']['Rate'][('', col)]
                    new_columns.append('Diff ' + col)
            column_order = column_order + new_columns
        csv = std_pivot.reindex(column_order, axis=1).to_csv(float_format='%.2f')
        csvToLatex(csv, key)

def getResultsForExperiment():
    dataframe_data = organizedResults(getExp=True)
    metricOrder = ['CER', 'Train CER', 'WER', 'Train WER', 'VUR', 'AUR', 'DUR', 'D VUR', 'D AUR']
    experiment_name = 'mvlrs_v1---cafe_10db---av_align---vad'
    df = pd.DataFrame(dataframe_data[experiment_name], columns=['CPS', 'Rate', 'Metric', 'ExpNR'])
    new_df = df[(df.Metric == 'CER') & (df.Rate > 30)].get(['CPS','ExpNR'])
    tuples = [tuple(x) for x in new_df.to_numpy()]
    new_new_df = df
    for t in tuples:
        print(df[(df.CPS == t[0]) & (df.ExpNR == t[1])].index)
        new_new_df = new_new_df.drop(df[(df.CPS == t[0]) & (df.ExpNR == t[1])].index)
    df_pivot = new_new_df.pivot_table(index='CPS', columns='Metric', values='Rate', aggfunc=np.mean)
    new_df_pivot = df_pivot[df_pivot['CER'] < 30]
    removeMetrics = ['Train CER', 'Train WER', 'WER']
    metrics = set([v[2] for v in df.values if v[2] not in removeMetrics])
    column_order = [metricOrder[i] for i in sorted([metricOrder.index(m) for m in metrics])]
    new_df_pivot = new_df_pivot.reindex(column_order, axis=1)
    
    old_experiment_name = experiment_name.replace('cafe_10db', 'clean')
    old_df = pd.DataFrame(dataframe_data[old_experiment_name], columns=['CPS', 'Rate', 'Metric', 'ExpNR'])
    old_df_pivot = old_df.pivot_table(index='CPS', columns='Metric', values='Rate', aggfunc=np.mean)
    
    for met in new_df_pivot.keys():
        print(met)
        new_df_pivot['Diff '+met] = ""
        for ind in new_df_pivot.index.values:
            print(ind)
            print('new', new_df_pivot[met][ind])
            print('old', old_df_pivot[met][ind])
            print('{:,.2f}'.format(new_df_pivot[met][ind] - old_df_pivot[met][ind]))
            new_df_pivot['Diff '+met][ind] = '{:,.2f}'.format(new_df_pivot[met][ind] - old_df_pivot[met][ind])
    csv = new_df_pivot.to_csv(float_format='%.2f')
    csvToLatex(csv, experiment_name)
    
def storePredictions():
    predictions = {}
    for config_file in os.listdir('./configs/finished/'):
        config = json.load(open('./configs/finished/' + config_file, 'r'))
        experiment_path = config['experiment_path']
        experiment_name = config['experiment_name']
        predictions[experiment_name] = {}
        prediction_path = './predictions/'+experiment_path+experiment_name+'/evaluateTrain/*'
        list_of_files = glob.glob(prediction_path)
        n_latest_file = max(list_of_files, key=os.path.getmtime)
        prediction_file = open(n_latest_file, 'r')
        lines = prediction_file.read().splitlines()
        for line in lines:
            line = line.replace(']','')
            split_line = line.split(' [')
            file = split_line[0].split(' ')[0]
            pred = ' '.join(split_line[0].split(' ')[1:])
            target = split_line[1]
            error = float(split_line[2])
            predictions[experiment_name][file] = {'pred':pred, 'target':target, 'error':error}
    p.dump(predictions, open('./predictions/predictions.p','wb'))
    
resultsToCSV()