import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import pandas as pd
from operator import itemgetter
from sys import platform

def hinton(matrix, savepath, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    #ax = ax if ax is not None else plt.gca()
    fig, ax = plt.subplots(2)

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax[0].patch.set_facecolor('gray')
    ax[0].set_aspect('equal', 'box')
    ax[0].xaxis.set_major_locator(plt.NullLocator())
    ax[0].yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax[0].add_patch(rect)

    ax[0].autoscale_view()
    ax[0].invert_yaxis()

    ax[1].plot(np.sum(matrix, axis=1))
    
    plt.savefig(savepath)
    
def visualize_loss(log_dir, file_name):
    batch_losses = []
    validation_losses = []
    with open(log_dir+file_name, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            if 'batch_loss' in l:
                bl = l.split(' ')[-1]
                batch_loss = float(bl)
                batch_losses.append(batch_loss)
            elif 'validation_loss' in l:
                bl = l.split(' ')[-1]
                validation_loss = float(bl)
                validation_losses.append(validation_loss)
    values = []
    for i in range(len(batch_losses)):
        values.append([batch_losses[i], validation_losses[i]])
    plt.plot(values)
    plt.legend(['batch_loss', 'validation_loss'])
    plt.show()
    
def visualize_measures():
    log_dir = "./logs/LRS3/"
    file_name = "av_to_chars_av_align_gru_v1b"
    values = []
    names = None
    with open(log_dir+file_name, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            if l.startswith('character'):
                if names is None:
                    names = [x for x in l.rsplit(' ') if "%" not in x][0:-1]
                measurements = [float(x.replace("%","")) for x in l.rsplit(' ') if "%" in x]
                values.append(measurements)
    plt.plot(values)
    plt.legend(names)
    plt.xticks([x for x in range(0,15,5)])
    plt.show()

def visualize_character_error_rates(log_dir, file_name):
    error_rates = [{}, {}, {}, {}, {}, {}, {}, {}]
    alt_error_rates = [{}, {}, {}, {}, {}, {}, {}, {}]
    with open(log_dir + file_name, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            if 'error_rate' in l:
                l = l.split(" ")
                for i in range(0,16,2):
                    name = l[i].replace(':','')
                    #print(name)
                    if not any([x in name for x in ['nD_', 'A_', 'V_']]):
                        val = float(l[i+1].replace('%',''))
                        if name in error_rates[i//2].keys():
                            error_rates[i//2][name].append(val)
                        else:
                            error_rates[i//2][name] = [val]
                    else:
                        val = float(l[i + 1].replace('%', ''))
                        if name in alt_error_rates[i // 2].keys():
                            alt_error_rates[i // 2][name].append(val)
                        else:
                            alt_error_rates[i // 2][name] = [val]
    for dict in error_rates:
        values = np.array(list(dict.values()))
        #values = values.reshape((values.shape[1], values.shape[0]))
        markers = ['D', 'o', 'v', 's', 'p']
        keys = list(dict.keys())
        for i,y in enumerate(values):
            plt.plot(y, marker=markers[i], linestyle=':', alpha=0.7)
        plt.legend(keys)
        plt.ylim(0, 100)
        plt.show()
    for dict in alt_error_rates:
        values = np.array(list(dict.values()))
        #values = values.reshape((values.shape[1], values.shape[0]))
        markers = ['D', 'o', 'v', 's', 'p']
        keys = list(dict.keys())
        for i,y in enumerate(values):
            plt.plot(y, marker=markers[i], linestyle=':', alpha=0.7)
        plt.legend(keys)
        plt.ylim(0, 100)
        plt.show()
    
def visualize_attention(attention_map):
    attn_len_y = attention_map.shape[0]
    attn_len_x = attention_map.shape[1]

    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(15, 10))
    ax = f.add_subplot(1, 1, 1)

    # Add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')

    # Add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    # Add labels
    ax.set_yticks(range(attn_len_y))

    ax.set_xticks(range(attn_len_x))
    
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()

    plt.show()
    
def visualize_all():
    log_dir = "./logs/mvlrs_v1/"
    file_name = f'exp{1}_a_skip_cps'
    #visualize_loss(log_dir, file_name)
    visualize_character_error_rates(log_dir, file_name)

def visualize_multiple_exp():
    log_dir = "./logs/mvlrs_v1/"
    experiments = []
    for exp_nr in [11,12,13,14,15]:
        experiments.append(f'exp{exp_nr}_a_skip_cps3')
    exp_values = {exp:{'cer':[], 'wer':[]} for exp in experiments}
    for experiment in experiments:
        with open(log_dir + experiment, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if 'error_rate' in line and not 'train' in line:
                    l = line.split(" ")
                    c_error_rate = float(l[1].replace('%',''))
                    w_error_rate = float(l[9].replace('%',''))
                    exp_values[experiment]['cer'].append(c_error_rate)
                    exp_values[experiment]['wer'].append(w_error_rate)
    for er in ['cer', 'wer']:
        all_values = []
        print(exp_values)
        for exp in exp_values:
            er_values = exp_values[exp][er]
            print(exp, len(er_values), er_values[-5:])
            all_values.append(er_values)
        markers = ['D', 'o', 'v', 's', 'p']
        keys = list(exp_values.keys())
        for i, y in enumerate(all_values):
            plt.plot(y, marker=markers[i], linestyle=':', alpha=0.7)
        plt.legend(keys)
        plt.ylim(0, 100)
        plt.show()
                    
def visualizeExperimentsBubble():
    movedpath = 'combined experiment/'
    path = './logs/mvlrs_v1/'+movedpath+'av_align/'
    for skip_layer in os.listdir(path):
        if skip_layer != 'n':
            data = []
            for i,cps in enumerate(os.listdir(path+skip_layer)):
                for layer_sizes in os.listdir(path+skip_layer+'/'+cps):
                    for exp in os.listdir(path+skip_layer+'/'+cps+'/'+layer_sizes):
                        file_path = path+skip_layer+'/'+cps+'/'+layer_sizes+'/'+exp
                        file = file_path.replace(movedpath,'').replace('/','_').replace('._logs_','')
                        file_path += '/'+file
                        with open(file_path, 'r') as f:
                            lines = f.read().splitlines()
                            error_rates = []
                            updated_states_rates = []
                            for line in lines:
                                if 'error_rate' in line and not 'train' in line:
                                    error_rate = float(line.split(' ')[1].replace('%',''))
                                    error_rates.append(error_rate)
                                if 'Updated_States_Rate' in line:
                                    usr = line.split(' ')[0].split(':')[1]
                                    updated_states_rates.append(float(usr)*100)
                            data.append([cps, error_rates[-1], 'error_rate'])
                            data.append([cps, updated_states_rates[-1], 'update_rate'])
            df = pd.DataFrame(data, columns=['cost_per_sample','value','value_type'])
            fig, ax = plt.subplots()
            plt.title(skip_layer)
            box_plot = sns.boxplot(y='value', x='cost_per_sample',data=df,palette='colorblind', hue='value_type', ax=ax)

            for i, artist in enumerate(ax.artists):
                # Set the linecolor on the artist to the facecolor, and set the facecolor to None
                col = artist.get_facecolor()
                artist.set_edgecolor(col)
                artist.set_facecolor('None')
    
                # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
                # Loop over them here, and use the same colour as above
                for j in range(i * 6, i * 6 + 6):
                    line = ax.lines[j]
                    line.set_color(col)
                    line.set_mfc(col)
                    line.set_mec(col)

            # Also fix the legend
            for legpatch in ax.get_legend().get_patches():
                col = legpatch.get_facecolor()
                legpatch.set_edgecolor(col)
                legpatch.set_facecolor('None')
            
            #c = 1
            #x_ticks = []
            #for res in results:
            #    bp = ax.boxplot(res, positions=[c,c+1], widths=0.6)
            #    x_ticks.append(c+0.5)
            #    c += 2
            #plt.xticks(x_ticks, x)
            #for xe, ze in zip(x, z):
            #    ax.scatter([xe]*len(ze), ze)
            #for i, txt in enumerate(y):
            #    #ax.text(x[i][0],z[i][0], str(txt)[:5])
            #    ax.annotate(str(txt)[:5] , (x[i],z[i][0]))
            plt.show()
                        
def visualizeSecondExperiment():
    movedpath = 'second experiment'
    path = './logs/mvlrs_v1/' + movedpath + '/av_align/'
    for skip_layer in os.listdir(path):
        if skip_layer != 'n':
            data = []
            for cps in os.listdir(path + skip_layer):
                for layer_sizes in os.listdir(path + skip_layer + '/' + cps):
                    for exp in os.listdir(path + skip_layer + '/' + cps + '/' + layer_sizes):
                        file_path = path + skip_layer + '/' + cps + '/' + layer_sizes + '/' + exp
                        file = file_path.replace(movedpath, '').replace('/', '_').replace('._logs_', '')
                        file_path += '/' + file
                        with open(file_path, 'r') as f:
                            lines = f.read().splitlines()
                            error_rates = []
                            updated_states_rates = []
                            for line in lines:
                                if 'error_rate' in line and not 'train' in line:
                                    error_rate = float(line.split(' ')[1].replace('%', ''))
                                    error_rates.append(error_rate)
                                if 'Updated_States_Rate' in line:
                                    usr = line.split(' ')[0].split(':')[1]
                                    updated_states_rates.append(float(usr) * 100)
                            data.append([cps, error_rates[-1], 'error_rate'])
                            data.append([cps, updated_states_rates[-1], 'update_rate'])
            df = pd.DataFrame(data, columns=['cost_per_sample', 'value', 'value_type'])
            fig, ax = plt.subplots()
            plt.title(skip_layer)
            box_plot = sns.boxplot(y='value', x='cost_per_sample', data=df, palette='colorblind', hue='value_type',
                                   ax=ax)

            for i, artist in enumerate(ax.artists):
                # Set the linecolor on the artist to the facecolor, and set the facecolor to None
                col = artist.get_facecolor()
                artist.set_edgecolor(col)
                artist.set_facecolor('None')

                # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
                # Loop over them here, and use the same colour as above
                for j in range(i * 6, i * 6 + 6):
                    line = ax.lines[j]
                    line.set_color(col)
                    line.set_mfc(col)
                    line.set_mec(col)

            # Also fix the legend
            for legpatch in ax.get_legend().get_patches():
                col = legpatch.get_facecolor()
                legpatch.set_edgecolor(col)
                legpatch.set_facecolor('None')

            plt.show()

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
#visualizeExperimentsBubble()
def getAllLogs(logsPath):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(logsPath):
        if platform == 'win32':
            dirpath = dirpath.replace('\\','/')
        if len(filenames) == 1:
            #f.append(dirpath+'/'+filenames[0])
            f.append([dirpath, filenames[0]])
    return f

def getParamsOfLog(logDir):
    #print(logDir)
    params = logDir.split('/')[4:]
    #print(params)
    return params

def formatCPS(cps):
    return format(eval(cps), 'f').rstrip('0').rstrip('.')

def getCPS(arch, skipped_layers, cps, one_line=False):
    print(cps)
    x = ['v','a','d']
    if len(cps) == 4:
        x = ['v', 'a','dv', 'da']
    if arch == 'bimodal' and skipped_layers == 'vad':
        y = [i for i in range(len(x))]
    else:
        y = [x.index(i) for i in skipped_layers]
    if len(skipped_layers) > 1:
        if one_line:
            ret = ' '.join([x[i]+':'+str(cps[i]) for i in y])
        else:
            ret = '\n'.join([x[i]+':'+str(cps[i]) for i in y])
    else:
        ret = '\n'.join([str(cps[i]) for i in y])
    return ret

def visualizeExperiment(architecture=['bimodal', 'av_align'], sl=['v','a','d','va','vd','ad','vad'], remove_metric='train_error_rate', save=False):
    for arch in architecture:
        print(arch)
        data = {x: [] for x in ['v', 'a', 'd', 'va', 'vd', 'ad', 'vad']}
        f = getAllLogs('./logs/mvlrs_v1/' + arch + '/')
        for logDir, logName in f:
            if arch == 'bimodal' and 'vad' in logDir:
                skipped_layers, v_cps, a_cps, d_v_cps, d_a_cps, _, _ = getParamsOfLog(logDir)
            else:
                skipped_layers, v_cps, a_cps, d_cps, _, _ = getParamsOfLog(logDir)
            v_cps = formatCPS(v_cps)
            a_cps = formatCPS(a_cps)
            if arch == 'bimodal' and 'vad' in logDir:
                d_v_cps = formatCPS(d_v_cps)
                d_a_cps = formatCPS(d_a_cps)
                cps = getCPS(arch, skipped_layers, [v_cps, a_cps, d_v_cps, d_a_cps])
            else:
                d_cps = formatCPS(d_cps)
                cps = getCPS(arch, skipped_layers, [v_cps, a_cps, d_cps])
            #print('CPS', cps)
            if cps != '1':
                with open(logDir+'/'+logName, 'r') as f:
                    lines = f.read().splitlines()
                    train_error_rates = []
                    error_rates = []
                    v_updated_states_rates = []
                    a_updated_states_rates = []
                    d_updated_states_rates = []
                    if arch == 'bimodal':
                        d_updated_states_rates_audio = []
                        d_updated_states_rates_video = []
                    for line in lines:
                        if 'error_rate' in line:
                            er = float(line.split(' ')[1].replace('%', ''))
                            if 'train' in line:
                                train_error_rates.append(er)
                            else:
                                error_rates.append(er)
                        if 'Video_Updated_States_Rate' in line:
                            v_usr = line.split(' ')[0].split(':')[1]
                            v_updated_states_rates.append(float(v_usr) * 100)
                        if 'Audio_Updated_States_Rate' in line:
                            a_usr = line.split(' ')[0].split(':')[1]
                            a_updated_states_rates.append(float(a_usr) * 100)
                        if 'Decoder_Updated_States_Rate' in line:
                            d_usr = float(line.split(' ')[0].split(':')[1])*100
                            if arch == 'bimodal':
                                if 'Audio' in line:
                                    d_updated_states_rates_audio.append(d_usr)
                                elif 'Video' in line:
                                    d_updated_states_rates_video.append(d_usr)
                            else:
                                d_updated_states_rates.append(d_usr)
                    if 'v' in skipped_layers:
                        data[skipped_layers].append([cps, train_error_rates[-1], 'train_error_rate'])
                        data[skipped_layers].append([cps, error_rates[-1], 'error_rate'])
                        data[skipped_layers].append([cps, v_updated_states_rates[-1], 'v_update_rate'])
                    if 'a' in skipped_layers:
                        data[skipped_layers].append([cps, train_error_rates[-1], 'train_error_rate'])
                        data[skipped_layers].append([cps, error_rates[-1], 'error_rate'])
                        data[skipped_layers].append([cps, a_updated_states_rates[-1], 'a_update_rate'])
                    if 'd' in skipped_layers:
                        data[skipped_layers].append([cps, train_error_rates[-1], 'train_error_rate'])
                        data[skipped_layers].append([cps, error_rates[-1], 'error_rate'])
                        if arch == 'bimodal':
                            data[skipped_layers].append([cps, d_updated_states_rates_audio[-1], 'd_update_rate_audio'])
                            data[skipped_layers].append([cps, d_updated_states_rates_video[-1], 'd_update_rate_video'])
                        else:
                            data[skipped_layers].append([cps, d_updated_states_rates[-1], 'd_update_rate'])
        for key, value in data.items():
            print(arch, key)
            print(value)
            if key in sl:
                value = sorted(value)
                if len(value) > 0:
                    sorter = ['error_rate', 'v_update_rate', 'a_update_rate', 'd_update_rate', 'd_update_rate_video', 'd_update_rate_audio']
                    sorterIndex = dict(zip(sorter, range(len(sorter))))
                    df = pd.DataFrame(value, columns=['CostPerSample', 'Rate', 'Metric'])
                    df['Metric'] = df['Metric'].map(sorterIndex)
                    df = df.sort_values(by=['CostPerSample','Metric'])
                    sorterIndex = dict(zip(range(len(sorter)), sorter))
                    df['Metric'] = df['Metric'].map(sorterIndex)

                    mean_df = pd.DataFrame(value, columns=['CostPerSample', 'Rate', 'Metric'])
                    line_data = mean_df.groupby(['CostPerSample', 'Metric']).agg({'Rate': 'mean'}).apply(list).to_dict()[
                        'Rate']
                    for k, v in line_data.items():
                        print(k[1], k[0].replace('\n',' '), v)

                    if 'v' in key:
                        plt.axhline(y=1.75, color='b')
                    if 'a' in key or 'd' in key:
                        plt.axhline(y=2.25, color='r')
                    
                    
                    df = df[df.Metric != remove_metric]
                    plot_data = [df]
                    if key == 'vad':
                        for cps in set(df.CostPerSample):
                            plot_data.append(df[df.CostPerSample == cps])
                        
                    for plot_d in plot_data:
                        fig, ax = plt.subplots()
                        if key == 'vad':
                            if len(list(set(plot_d.CostPerSample))) == 1:
                                info = list(set(plot_d.CostPerSample))[0]
                                remove = ['\n', 'v:', 'a:', 'd:']
                                for r in remove:
                                    info = info.replace(r, '')
                                info = '_'+info
                            else:
                                info = ''
                        else:
                            info = ''
                        plt.title(arch+'_'+key+info)
                        box_plot = sns.boxplot(y='Rate', x='CostPerSample', data=plot_d, palette='colorblind', hue='Metric',
                                               ax=ax)
                        
                        for i, artist in enumerate(ax.artists):
                            # Set the linecolor on the artist to the facecolor, and set the facecolor to None
                            col = artist.get_facecolor()
                            artist.set_edgecolor(col)
                            artist.set_facecolor('None')
    
                            # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
                            # Loop over them here, and use the same colour as above
                            for j in range(i * 6, i * 6 + 6):
                                line = ax.lines[j]
                                line.set_color(col)
                                line.set_mfc(col)
                                line.set_mec(col)
    
                        # Also fix the legend
                        for legpatch in ax.get_legend().get_patches():
                            col = legpatch.get_facecolor()
                            legpatch.set_edgecolor(col)
                            legpatch.set_facecolor('None')
    
                        if save:
                            plt.savefig('./visualization/'+arch+'/boxplot_allCPS_'+key+info+'.png', dpi=320)
                        else:
                            plt.show()
                        plt.close(fig)

def moveFirstExperimentLogs():
    f = getAllLogs('./logs/mvlrs_v1/first experiment/')
    for logDir, logName in f:
        if not '_n_' in logName:
            old_logDir = logDir
            old_logName = logName
            architecture, skipped_layers, cps, number_of_layers, exp_number = getParamsOfLog(logDir)
            new_cps = float(cps[0]+'.'+cps[1:])
            if 'v' in skipped_layers:
                logDir = logDir.replace(cps,format_e(new_cps)+'/0E+00'+'/0E+00').replace('first experiment','combined experiment')
                logName = logName.replace(cps,format_e(new_cps)+'_0E+00'+'_0E+00')
            if 'a' in skipped_layers:
                logDir = logDir.replace(cps, '0E+00/'+format_e(new_cps) + '/0E+00').replace('first experiment','combined experiment')
                logName = logName.replace(cps, '0E+00_'+format_e(new_cps) + '_0E+00')
            if 'd' in skipped_layers:
                logDir = logDir.replace(cps, '0E+00' + '/0E+00/'+format_e(new_cps)).replace('first experiment','combined experiment')
                logName = logName.replace(cps,'0E+00' + '_0E+00_' + format_e(new_cps))
            print(logDir)
            print(logName)
            from shutil import copyfile
            os.makedirs(logDir+'/')
            copyfile(old_logDir+'/'+old_logName, logDir+'/'+logName)

def visualizeCombinedExperiments():
    f = getAllLogs('./logs/mvlrs_v1/combined experiment/')
    data = {x: {} for x in ['v', 'a', 'd', 'va', 'vd', 'ad', 'vad']}
    for logDir, logName in f:
        architecture, skipped_layers, v_cps, a_cps, d_cps, number_of_layers, exp_number = getParamsOfLog(logDir)
        v_cps = formatCPS(v_cps)
        a_cps = formatCPS(a_cps)
        d_cps = formatCPS(d_cps)
        with open(logDir + '/' + logName, 'r') as f:
            lines = f.read().splitlines()
            error_rates = []
            v_updated_states_rates = []
            a_updated_states_rates = []
            d_updated_states_rates = []
            for line in lines:
                if 'error_rate' in line and not 'train' in line:
                    error_rate = float(line.split(' ')[1].replace('%', ''))
                    error_rates.append(error_rate)
                if 'Video_Updated_States_Rate' in line:
                    v_usr = line.split(' ')[0].split(':')[1]
                    v_updated_states_rates.append(float(v_usr) * 100)
                if 'Audio_Updated_States_Rate' in line:
                    a_usr = line.split(' ')[0].split(':')[1]
                    a_updated_states_rates.append(float(a_usr) * 100)
                if 'Decoder_Updated_States_Rate' in line:
                    d_usr = line.split(' ')[0].split(':')[1]
                    d_updated_states_rates.append(float(d_usr) * 100)
            cps = getCPS(architecture, skipped_layers, [v_cps, a_cps, d_cps])
            # print(cps)
            if not cps in data[skipped_layers].keys():
                data[skipped_layers][cps] = []
            if 'v' in skipped_layers:
                data[skipped_layers][cps].append([error_rates[-1], 'Error Rate'])
                data[skipped_layers][cps].append([v_updated_states_rates[-1], 'Video Update Rate'])
            if 'a' in skipped_layers:
                data[skipped_layers][cps].append([error_rates[-1], 'Error Rate'])
                data[skipped_layers][cps].append([a_updated_states_rates[-1], 'Audio Update Rate'])
            if 'd' in skipped_layers:
                data[skipped_layers][cps].append([error_rates[-1], 'Error Rate'])
                data[skipped_layers][cps].append([d_updated_states_rates[-1], 'Decoder Update Rate'])
    data_combi = {key:value for key,value in data.items() if len(key) > 1}
    data_single = {key:value for key,value in data.items() if len(key) == 1}

    for key, cps_dict in data_combi.items():
        for cps, value in cps_dict.items():
            value = [[cps, x[0], x[1]] for x in value]
            print(cps)
            cps_values = cps.split('\n')
            cps_values = {x.split(':')[0]: x.split(':')[1] for x in cps_values}
            print(key)
            for skipped_layer in key:
                print(cps_values)
                skipped_layer_values = data_single[skipped_layer][cps_values[skipped_layer]]
                for skipped_layer_value in skipped_layer_values:
                    value.append([skipped_layer+':'+cps_values[skipped_layer], skipped_layer_value[0], skipped_layer_value[1]])

            df = pd.DataFrame(value, columns=['cost_per_sample', 'value', 'Metric'])
            fig, ax = plt.subplots()

            title = key.replace('d','Decoder').replace('v','Video').replace('a','Audio') + ' Combination'
            plt.title(title)
            print(df)
            my_pal = {"Video Update Rate": 'blue', "Audio Update Rate": "green", "Decoder Update Rate": "black", "Error Rate":"red"}
            box_plot = sns.boxplot(y='value', x='cost_per_sample', data=df, palette='bright', hue='Metric',
                                   ax=ax, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"5"})

            for i, artist in enumerate(ax.artists):
                # Set the linecolor on the artist to the facecolor, and set the facecolor to None
                col = artist.get_facecolor()
                artist.set_edgecolor(col)
                #artist.set_facecolor('None')

                # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
                # Loop over them here, and use the same colour as above
                for j in range(i * 6, i * 6 + 6):
                    line = ax.lines[j]
                    line.set_color(col)
                    line.set_mfc(col)
                    line.set_mec(col)

            # Also fix the legend
            for legpatch in ax.get_legend().get_patches():
                col = legpatch.get_facecolor()
                legpatch.set_edgecolor(col)
                #legpatch.set_facecolor('None')

            #if 'v' in key:
            #    plt.axhline(y=1.75, color='b')
            #if 'a' in key or 'd' in key:
            #    plt.axhline(y=2.25, color='r')

            plt.xlabel('Cost per Sample')
            plt.ylabel('Rate')
            plt.show()
            #cps_fn = '_'.join(cps_values.values()).replace('.', '')
            #plt.savefig('./visualization/boxplot_compare_' + key + '_' + cps_fn + '.png', dpi=320)

def visualizeCombinedExperiments2(architecture=['bimodal'], save=False):
    for arch in architecture:
        print(arch)
        f = getAllLogs('./logs/mvlrs_v1/'+arch+'/')
        data = {x: {} for x in ['v', 'a', 'd', 'va', 'vd', 'ad', 'vad']} #['v', 'a', 'd', 'va', 'vd', 'ad', 'vad']
        for logDir, logName in f:
            if arch == 'bimodal':
                skipped_layers, v_cps, a_cps, d_v_cps, d_a_cps, _, _ = getParamsOfLog(logDir)
            else:
                skipped_layers, v_cps, a_cps, d_cps, _, _ = getParamsOfLog(logDir)
            v_cps = formatCPS(v_cps)
            a_cps = formatCPS(a_cps)
            if arch == 'bimodal':
                d_v_cps = formatCPS(d_v_cps)
                d_a_cps = formatCPS(d_a_cps)
                cps = getCPS(arch, skipped_layers, [v_cps, a_cps, d_v_cps, d_a_cps], True)
            else:
                d_cps = formatCPS(d_cps)
                cps = getCPS(arch, skipped_layers, [v_cps, a_cps, d_cps], True)
            print('cps', cps)
            
            with open(logDir + '/' + logName, 'r') as f:
                lines = f.read().splitlines()
                error_rates = []
                train_error_rates = []
                v_updated_states_rates = []
                a_updated_states_rates = []
                if arch == 'bimodal':
                    d_updated_states_rates_video = []
                    d_updated_states_rates_audio = []
                else:
                    d_updated_states_rates = []
                for line in lines:
                    if 'error_rate' in line:
                        er = float(line.split(' ')[1].replace('%', ''))
                        if 'train' in line:
                            train_error_rates.append(er)
                        else:
                            error_rates.append(er)
                    if 'Video_Updated_States_Rate' in line:
                        v_usr = line.split(' ')[0].split(':')[1]
                        v_updated_states_rates.append(float(v_usr) * 100)
                    if 'Audio_Updated_States_Rate' in line:
                        a_usr = line.split(' ')[0].split(':')[1]
                        a_updated_states_rates.append(float(a_usr) * 100)
                    if 'Decoder_Updated_States_Rate' in line:
                        d_usr = float(line.split(' ')[0].split(':')[1]) * 100
                        if arch == 'bimodal':
                            if 'Audio' in line:
                                d_updated_states_rates_audio.append(d_usr)
                            elif 'Video' in line:
                                d_updated_states_rates_video.append(d_usr)
                        else:
                            d_updated_states_rates.append(d_usr)
                if not cps in data[skipped_layers].keys():
                    data[skipped_layers][cps] = []
                if 'v' in skipped_layers:
                    data[skipped_layers][cps].append([train_error_rates[-1], 'Train Error Rate'])
                    data[skipped_layers][cps].append([error_rates[-1], 'Error Rate'])
                    data[skipped_layers][cps].append([v_updated_states_rates[-1], 'Video Update Rate'])
                if 'a' in skipped_layers:
                    data[skipped_layers][cps].append([train_error_rates[-1], 'Train Error Rate'])
                    data[skipped_layers][cps].append([error_rates[-1], 'Error Rate'])
                    data[skipped_layers][cps].append([a_updated_states_rates[-1], 'Audio Update Rate'])
                if 'd' in skipped_layers:
                    data[skipped_layers][cps].append([cps, train_error_rates[-1], 'Train Error Rate'])
                    data[skipped_layers][cps].append([cps, error_rates[-1], 'Error Rate'])
                    if arch == 'bimodal':
                        data[skipped_layers][cps].append([cps, d_updated_states_rates_audio[-1], 'Decoder Audio Update Rate'])
                        data[skipped_layers][cps].append([cps, d_updated_states_rates_video[-1], 'Decoder Video Update Rate'])
                    else:
                        data[skipped_layers][cps].append([cps, d_updated_states_rates[-1], 'Decoder Update Rate'])
        data_combi = {key:value for key,value in data.items() if len(key) > 1}
        data_single = {key:value for key,value in data.items() if len(key) == 1}
        print(data_single)
        exit()
    
        for key, cps_dict in data_combi.items():
            for cps, value in cps_dict.items():
                value = [[cps, x[0], x[1]] for x in value]
                mean_values = []
                print('cps', cps)
                cps_values = cps.split(' ')
                cps_values = {x.split(':')[0]: x.split(':')[1] for x in cps_values}
                print(key)
                for skipped_layer in key:
                    print(cps_values)
                    skipped_layer_values = data_single[skipped_layer][cps_values[skipped_layer]]
                    for skipped_layer_value in skipped_layer_values:
                        mean_values.append([skipped_layer+':'+cps_values[skipped_layer], float(skipped_layer_value[0]), skipped_layer_value[1]])
                df = pd.DataFrame(value, columns=['cost_per_sample', 'value', 'Metric'])
                print(df)
                mean_df = pd.DataFrame(mean_values, columns=['cost_per_sample', 'value', 'Metric'])
                print(mean_df)
                fig, ax = plt.subplots()
                line_data = mean_df.groupby(['cost_per_sample','Metric']).agg({'value':'mean'}).apply(list).to_dict()['value']
                print(line_data)
    
                colors = sns.color_palette('bright')
    
                if 'v' in key:
                    plt.axhline(y=line_data[('v:' + cps_values['v'], 'Video Update Rate')], color=colors[1], label='(old) Video Update Rate', linewidth=1.5)
                    plt.axhline(y=line_data[('v:' + cps_values['v'], 'Error Rate')], color=colors[1], linestyle='dashdot', label='(old) Video Error Rate', linewidth=1.5)
                if 'a' in key:
                    plt.axhline(y=line_data[('a:' + cps_values['a'], 'Audio Update Rate')], color=colors[2], label='(old) Audio Update Rate', linewidth=1.5)
                    plt.axhline(y=line_data[('a:' + cps_values['a'], 'Error Rate')], color=colors[2], linestyle='dashed', label='(old) Audio Error Rate', linewidth=1.5)
                if 'd' in key:
                    if arch == 'bimodal':
                        plt.axhline(y=line_data[('d:' + cps_values['d'], 'Error Rate')], color=colors[3],
                                    linestyle='dotted', label='(old) Decoder Update Rate', linewidth=1.5)
                        plt.axhline(y=line_data[('d:' + cps_values['d'], 'Decoder Update Rate')], color=colors[3],
                                    label='(old) Decoder Error Rate', linewidth=1.5)
                    else:
                        plt.axhline(y=line_data[('d:' + cps_values['d'], 'Error Rate')], color=colors[3], linestyle='dotted', label='(old) Decoder Update Rate', linewidth=1.5)
                        plt.axhline(y=line_data[('d:' + cps_values['d'], 'Decoder Update Rate')], color=colors[3], label='(old) Decoder Error Rate', linewidth=1.5)
    
                print(df)
                box_plot = sns.boxplot(y='value', x='cost_per_sample', data=df[df.Metric != 'Train Error Rate'], palette='bright', hue='Metric',
                                       ax=ax, showmeans=True, meanprops={"marker":"o",
                           "markerfacecolor":"white",
                           "markeredgecolor":"black",
                          "markersize":"5"},
                                       boxprops=dict(alpha=.3))

                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                           ncol=2, mode="expand", borderaxespad=0.)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                print(len(ax.lines))
                for i, artist in enumerate(ax.artists):
                    print('test', i)
                    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
                    #col = artist.get_facecolor()
                    col = colors[i]
                    artist.set_edgecolor(col)
                    #artist.set_facecolor('None')
    
                    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
                    # Loop over them here, and use the same colour as above
                    for j in range(i * 6 + len(key)*2, i * 6 + 6 + len(key)*2):
                        line = ax.lines[j]
                        line.set_color(col)
                        line.set_mfc(col)
                        line.set_mec(col)
    
                # Also fix the legend
                for legpatch in ax.get_legend().get_patches():
                    col = legpatch.get_facecolor()
                    legpatch.set_edgecolor(col)
                    #legpatch.set_facecolor('None')
    
                #if 'v' in key:
                #    plt.axhline(y=1.75, color='b')
                #if 'a' in key or 'd' in key:
                #    plt.axhline(y=2.25, color='r')
                title = key.replace('d', 'Decoder').replace('v', 'Video').replace('a', 'Audio') + ' Combination'
                plt.title(arch+'_'+title)
                plt.xlabel('Cost per Sample')
                plt.ylabel('Percentage')
                if save:
                    cps_fn = '_'.join(cps_values.values()).replace('.','')
                    plt.savefig('./visualization/'+arch+'/boxplot_compareLines_' + key + '_' + cps_fn + '.png', dpi=320)
                else:
                    plt.show()
                
def sortCPSList(lst):
    if ' ' in lst[0]:
        l = [x.split(' ')[0] for x in lst]
        m = [x.split(' ')[0] for x in lst]
        l.sort(key=float)
        new_lst = [lst[m.index(i)] for i in l]
        lst = new_lst
    else:
        lst.sort(key=float)
    return lst

#visualizeCombinedExperiments()
#visualizeCombinedExperiments2(save=True)
#visualizeExperiment(save=True)

def visualizeResults(alpha=0.5, save=False):
    from utilities import organizedResults
    dataframe_data = organizedResults()
    dataframes_single = {}
    markers = ['o', '^', 's', 'p', 'p', '*', 'x']
    colors = ['blue','red','orange','yellow','grey','brown','green']
    for metric in ['CER', 'WER']:
        for key in [key for key in dataframe_data.keys() if not 'va' in key and not key.endswith('n')]:
            arch ='---'.join(key.split('---')[:-1])
            os.makedirs('./visualization/'+metric+'/'+arch+'/', exist_ok=True)
            print(key)
            sorter = [metric, 'VUR', 'AUR', 'DUR', 'D VUR',
                      'D AUR']
            sorterIndex = dict(zip(sorter, range(len(sorter))))
            df = pd.DataFrame(dataframe_data[key], columns=['CostPerSample', 'Rate', 'Metric'])
            df['Metric'] = df['Metric'].map(sorterIndex)
            df = df.sort_values(by=['CostPerSample', 'Metric'])
            sorterIndex = dict(zip(range(len(sorter)), sorter))
            df['Metric'] = df['Metric'].map(sorterIndex)
            dataframes_single[key] = df.groupby(['CostPerSample','Metric']).agg({'Rate':'mean'}).apply(list).to_dict()['Rate']
            
            fig, ax = plt.subplots()
            colors = sns.color_palette('bright')
            #box_plot = sns.boxplot(y='Rate', x='CostPerSample', data=df[df.Metric != 'Train'+metric],
            #                       palette='bright', hue='Metric',
            #                       ax=ax, showmeans=True, meanprops={"marker": "o",
            #                                                         "markerfacecolor": "white",
            #                                                         "markeredgecolor": "black",
            #                                                         "markersize": "5"},
            #                       boxprops=dict(alpha=alpha))
            lst = list(df['CostPerSample'].unique())
            lst = sortCPSList(lst)
            
            scatter_plot = sns.stripplot(y='Rate', x='CostPerSample', data=df[df.Metric != 'Train'+metric],
                                   palette='bright', hue='Metric',
                                   ax=ax, order=lst)
            
            for i, artist in enumerate(ax.artists):
                # Set the linecolor on the artist to the facecolor, and set the facecolor to None
                col = artist.get_facecolor()
                # col = colors[i//6]
                artist.set_edgecolor(col)
                # artist.set_facecolor('None')
        
                # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
                # Loop over them here, and use the same colour as above
                for j in range(i * 7, i * 7 + 7):
                    line = ax.lines[j]
                    line.set_color(col)
                    line.set_mfc(col)
                    line.set_mec(col)
            
            avsr_error_key = key.split('---')
            avsr_error_key[-1] = 'n'
            avsr_error_key = '---'.join(avsr_error_key)
            avsr_error = np.mean([x[1] for x in dataframe_data[avsr_error_key] if x[2] == metric])
            
            plt.axhline(y=avsr_error, color=colors[0],
                        linestyle='dashed', alpha=alpha,
                        label='AVSR-tf1'+metric, linewidth=1.5)

            lgd = plt.legend()
            #lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            #          fancybox=True, shadow=True, ncol=5)
            # Also fix the legend
            #for legpatch in ax.get_legend().get_patches():
            #    col = legpatch.get_facecolor()
            #    legpatch.set_edgecolor(col)
            #    # legpatch.set_facecolor('None')
            dataset, noise, architecture, skip_layer = key.split('---')
            title = dataset.replace('mvlrs_v1','LRS2') + ' ' + noise + ' ' + architecture + ' ' + skip_layer
            plt.title(title.upper())
            plt.xlabel('Cost per Sample')
            plt.ylabel('Rate')
            plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            if architecture == 'bimodal':
                if skip_layer == 'd':
                    locs, ticks = plt.xticks()
                    new_ticks = []
                    for tick in ticks:
                        new_ticks.append('V:'+tick.get_text().replace(' ', '\nA:'))
                    plt.xticks(range(len(new_ticks)), new_ticks)
                if skip_layer == 'va':
                    locs, ticks = plt.xticks()
                    new_ticks = []
                    for tick in ticks:
                        new_ticks.append('V:' + tick.get_text().replace(' ', ' A:'))
                    plt.xticks(range(len(new_ticks)), new_ticks)
                if skip_layer == 'vad':
                    locs, ticks = plt.xticks()
                    new_ticks = []
                    for tick in ticks:
                        cpsv, cpsa, cpsdv, cpsda = tick.get_text().split(' ')
                        new_ticks.append('V:'+cpsv+' A:'+cpsa+' DV:'+cpsa+' DA:'+cpsa)
                    plt.xticks(range(len(new_ticks)), new_ticks)
            ax.grid(axis='y', which='major', linestyle='-', linewidth='0.5', color='gray')
            ax.grid(axis='y', which='minor', linestyle=':', linewidth='0.5', color='black')
            #plt.tight_layout()
            if save:
                plt.savefig('./visualization/'+metric+'/' + arch + '/boxplot_allCPS_' + key.split('---')[-1] + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=320)
            else:
                plt.show()
            plt.close(fig)
        
        for key in [key for key in dataframe_data.keys() if 'va' in key]:
            if 'LRS3' not in key:
                arch = '---'.join(key.split('---')[:-1])
                os.makedirs('./visualization/'+metric+'/' + arch + '/', exist_ok=True)
                print(key)
                print('key', key)
                sorter = [metric, 'VUR', 'AUR', 'DUR', 'D VUR',
                          'D AUR']
                sorterIndex = dict(zip(sorter, range(len(sorter))))
                df = pd.DataFrame(dataframe_data[key], columns=['CostPerSample', 'Rate', 'Metric'])
                
                for cps in df['CostPerSample'].unique():
                    print('cps', cps)
                    single_df = df[df['CostPerSample'] == cps]
                    fig, ax = plt.subplots()
                    colors = sns.color_palette('bright')
                    if metric == "CER":
                        remove_metric = "WER"
                    else:
                        remove_metric = "CER"
                    
                    single_df_removed_metric = single_df[single_df.Metric != 'Train '+remove_metric]
                    single_df_removed_metric = single_df_removed_metric[single_df_removed_metric.Metric != 'Train '+metric]
                    single_df_removed_metric = single_df_removed_metric[single_df_removed_metric.Metric != remove_metric]
                    
                    #box_plot = sns.boxplot(y='Rate', x='CostPerSample', data=single_df_removed_metric,
                    #                       palette='bright', hue='Metric',
                    #                       ax=ax, showmeans=True, meanprops={"marker": "o",
                    #                                                         "markerfacecolor": "white",
                    #                                                         "markeredgecolor": "black",
                    #                                                         "markersize": "5"},
                    #                       boxprops=dict(alpha=alpha))
                    
                    
                    lst = list(single_df_removed_metric['CostPerSample'].unique())
                    lst = sortCPSList(lst)
                    
                    stripplot = sns.stripplot(y='Rate', x='CostPerSample', data=single_df_removed_metric,
                                              marker=markers[0], hue='Metric',palette='bright',
                                              ax=ax, order=lst)

                    #for i, artist in enumerate(ax.artists):
                    #    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
                    #    col = artist.get_facecolor()
                    #    # col = colors[i]
                    #    artist.set_edgecolor(col)
                    #    # artist.set_facecolor('None')
            
                    #    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
                    #    # Loop over them here, and use the same colour as above
                    #    for j in range(i * 7, i * 7 + 7):
                    #        line = ax.lines[j]
                    #        line.set_color(col)
                    #        line.set_mfc(col)
                    #        line.set_mec(col)
                    
                    col_ind = {met:h for h, met in enumerate(single_df_removed_metric['Metric'].unique())}
                    colors = sns.color_palette('bright')
                    
                    if 'bimodal' in key:
                        cps_values = [0.0, 0.0, 0.0, 0.0]
                        l = key.split('---')[-1]
                        indices = ['vad'.index(c) for c in l]
                        for x, ind in enumerate(indices): cps_values[ind] = cps.split(' ')[x]
                        if l == 'vad': cps_values[3] = cps.split(' ')[-1]
                        v_cps, a_cps, dv_cps, da_cps = cps_values
                    else:
                        cps_values = [0.0, 0.0, 0.0]
                        l = key.split('---')[-1]
                        indices = ['vad'.index(c) for c in l]
                        for x, ind in enumerate(indices): cps_values[ind] = cps.split(' ')[x]
                        v_cps, a_cps, d_cps = cps_values
                    if 'v' in key.split('---')[-1]:
                        if 'clean' not in key:
                            if 'cafe' in key or 'zeroing' in key:
                                single_key = '---'.join(key.split('---')[2:-1]) + '---v'
                            elif 'targeted' in key:
                                single_key = '---'.join(key.split('---')[2:-1]) + '---v'
                            single_key = key.split('---')[0]+'---clean---' + single_key
                        else:
                            single_key = '---'.join(key.split('---')[:-1]) + '---v'
                        plt.axhline(y=dataframes_single[single_key][(v_cps, 'VUR')], color=colors[col_ind['VUR']],
                                    linestyle='dotted', alpha=alpha,
                                    label='(old) VUR', linewidth=1.5)
                        plt.axhline(y=dataframes_single[single_key][(v_cps, metric)], color=colors[col_ind[metric]],
                                    linestyle='dashed',alpha=alpha,
                                    label='(old) V '+metric, linewidth=1.5)
                    if 'a' in key.split('---')[-1]:
                        if 'clean' not in key:
                            if 'cafe' in key or 'zeroing' in key:
                                single_key = '---'.join(key.split('---')[2:-1]) + '---a'
                            elif 'targeted' in key:
                                single_key = '---'.join(key.split('---')[2:-1]) + '---a'
                            single_key = key.split('---')[0]+'---clean---' + single_key
                        else:
                            single_key = '---'.join(key.split('---')[:-1]) + '---a'
                        plt.axhline(y=dataframes_single[single_key][(a_cps, 'AUR')], color=colors[col_ind['AUR']],
                                    linestyle='dotted', alpha=alpha,
                                    label='(old) AUR', linewidth=1.5)
                        plt.axhline(y=dataframes_single[single_key][(a_cps, metric)], color=colors[col_ind[metric]],
                                    linestyle='dashed', alpha=alpha,
                                    label='(old) A '+metric, linewidth=1.5)
                    if 'd' in key.split('---')[-1]:
                        if 'clean' not in key:
                            if 'cafe' in key or 'zeroing' in key:
                                single_key = '---'.join(key.split('---')[2:-1]) + '---d'
                            elif 'targeted' in key:
                                single_key = '---'.join(key.split('---')[2:-1]) + '---d'
                            single_key = key.split('---')[0]+'---clean---' + single_key
                        else:
                            single_key = '---'.join(key.split('---')[:-1]) + '---d'
                        if 'bimodal' in key:
                            plt.axhline(y=dataframes_single[single_key][(dv_cps+' '+dv_cps.replace('DV','DA'), 'D VUR')], color=colors[col_ind['D VUR']],
                                        linestyle='dotted', alpha=alpha,
                                        label='(old) D VUR', linewidth=1.5)
                            plt.axhline(y=dataframes_single[single_key][(da_cps.replace('DA','DV')+' '+da_cps, 'D AUR')], color=colors[col_ind['D AUR']],
                                        linestyle='dotted', alpha=alpha,
                                        label='(old) D AUR', linewidth=1.5)
                            plt.axhline(y=dataframes_single[single_key][(dv_cps+' '+dv_cps.replace('DV','DA'), metric)], color=colors[col_ind[metric]],
                                        linestyle='dashed', alpha=alpha,
                                        label='(old) D V '+metric, linewidth=1.5)
                            plt.axhline(y=dataframes_single[single_key][(da_cps.replace('DA','DV')+' '+da_cps, metric)], color=colors[col_ind[metric]],
                                        linestyle='dashed', alpha=alpha,
                                        label='(old) D A '+metric, linewidth=1.5)
                        else:
                            plt.axhline(y=dataframes_single[single_key][(d_cps, 'DUR')], color=colors[col_ind['DUR']],
                                        linestyle='dotted', alpha=alpha,
                                        label='(old) DUR', linewidth=1.5)
                            plt.axhline(y=dataframes_single[single_key][(d_cps, metric)], color=colors[col_ind[metric]],
                                        linestyle='dashed', alpha=alpha,
                                        label='(old) D '+metric, linewidth=1.5)
                            
                    lgd = plt.legend()
                    # Also fix the legend
                   # for legpatch in ax.get_legend().get_patches():
                        #col = legpatch.get_facecolor()
                        #legpatch.set_edgecolor(col)
                        # legpatch.set_facecolor('None')
    
                    dataset, noise, architecture, skip_layer = key.split('---')
                    title = dataset.replace('mvlrs_v1','LRS2') + ' ' + noise + ' ' + architecture + ' ' + skip_layer
                    plt.title(title.upper())
                    plt.xlabel('Cost per Sample')
                    plt.ylabel('Percentage')
                    plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
                    
                    ax.grid(axis='y', which='major', linestyle='-', linewidth='0.5', color='gray')
                    ax.grid(axis='y', which='minor', linestyle=':', linewidth='0.5', color='black')
                    plt.tight_layout()
                    if save:
                        cps = cps.replace(':','').replace('V','').replace('A','').replace('D','').replace(' ', '_')
                        plt.savefig('./visualization/'+metric+ '/' + arch + '/boxplot_compareLines_' + key + '_' + cps + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=320)
                    else:
                        plt.show()
                    plt.close()

def showImagesHorizontally():
    from matplotlib.image import imread
    path = 'D:/Users/Fabio/Downloads/captures/'
    for folder in os.listdir(path):
        if not 'lips' in folder:
            list_of_files = os.listdir(path+folder)
            #list_of_files = [x for i,x in enumerate(list_of_files) if i%4 == 0]
            list_of_files = list_of_files[:16]
            fig = plt.figure()
            number_of_files = len(list_of_files)
            print(number_of_files)
            for i in range(number_of_files):
                a=fig.add_subplot(2,number_of_files,i+1)
                image = plt.imread(path+folder+'/'+list_of_files[i])
                plt.imshow(image,cmap='Greys_r')
                plt.axis('off')
            list_of_files = os.listdir(path + folder + '-lips')[:16]
            number_of_files = len(list_of_files)
            for i in range(number_of_files):
                a=fig.add_subplot(1,number_of_files,i+1)
                image = plt.imread(path+folder+'-lips'+'/'+list_of_files[i])
                plt.imshow(image,cmap='Greys_r')
                plt.axis('off')
            plt.show()

def showFaces():
    path = 'D:/Users/Fabio/Downloads/captures/'
    files = {}
    fig, axs = plt.subplots(5,16)
    for j,folder in enumerate(os.listdir(path)):
        if 'lips' in folder:
            list_of_files = os.listdir(path + folder)
            list_of_files = list_of_files[:16]
            files[folder] = list_of_files
    for j, keyval in enumerate(files.items()):
        folder, list_of_files = keyval
        number_of_files = len(list_of_files)
        for i in range(number_of_files):
            image = plt.imread(path + folder + '/' + list_of_files[i])
            axs[j,i].imshow(image)
            axs[j,i].axis('off')
    plt.show()

visualizeResults(save=True)