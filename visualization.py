import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import pandas as pd

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
        if len(filenames) == 1:
            #f.append(dirpath+'/'+filenames[0])
            f.append([dirpath, filenames[0]])
    return f

def getParamsOfLog(logDir):
    params = logDir.split('/')[4:]
    return params

def formatCPS(cps):
    return format(eval(cps), 'f').rstrip('0').rstrip('.')

def getCPS(skipped_layers, cps, one_line=False):
    x = ['v','a','d']
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
        f = getAllLogs('./logs/mvlrs_v1/combined experiment/' + arch + '/')
        for logDir, logName in f:
            _, skipped_layers, v_cps, a_cps, d_cps, _, _ = getParamsOfLog(logDir)
            v_cps = formatCPS(v_cps)
            a_cps = formatCPS(a_cps)
            d_cps = formatCPS(d_cps)
            cps = getCPS(skipped_layers, [v_cps, a_cps, d_cps])
            # print(cps)
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
            print(value)
            print(key)
            if key in sl:
                value = sorted(value)
                if len(value) > 0:
                    df = pd.DataFrame(value, columns=['Cost Per Sample', 'Rate', 'Metric'])

                    mean_df = pd.DataFrame(value, columns=['Cost Per Sample', 'Rate', 'Metric'])
                    line_data = mean_df.groupby(['Cost Per Sample', 'Metric']).agg({'Rate': 'mean'}).apply(list).to_dict()[
                        'Rate']
                    for k, v in line_data.items():
                        print(k[1], k[0], v)

                    fig, ax = plt.subplots()
                    #print(df)
                    if 'v' in key:
                        plt.axhline(y=1.75, color='b')
                    if 'a' in key or 'd' in key:
                        plt.axhline(y=2.25, color='r')

                    plt.title(arch+'_'+key)
                    box_plot = sns.boxplot(y='Rate', x='Cost Per Sample', data=df[df.Metric != remove_metric], palette='colorblind', hue='Metric',
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
                        plt.savefig('./visualization/'+arch+'/boxplot_allCPS_'+key+'.png', dpi=320)
                    else:
                        plt.show()

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
            cps = getCPS(skipped_layers, [v_cps, a_cps, d_cps])
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
            plt.ylabel('Percentage')
            plt.show()
            #cps_fn = '_'.join(cps_values.values()).replace('.', '')
            #plt.savefig('./visualization/boxplot_compare_' + key + '_' + cps_fn + '.png', dpi=320)

def visualizeCombinedExperiments2():
    f = getAllLogs('./logs/mvlrs_v1/combined experiment/')
    data = {x: {} for x in ['v', 'a', 'd', 'va', 'vd', 'ad', 'vad']} #['v', 'a', 'd', 'va', 'vd', 'ad', 'vad']
    for logDir, logName in f:
        architecture, skipped_layers, v_cps, a_cps, d_cps, number_of_layers, exp_number = getParamsOfLog(logDir)
        v_cps = formatCPS(v_cps)
        a_cps = formatCPS(a_cps)
        d_cps = formatCPS(d_cps)
        with open(logDir + '/' + logName, 'r') as f:
            lines = f.read().splitlines()
            error_rates = []
            train_error_rates = []
            v_updated_states_rates = []
            a_updated_states_rates = []
            d_updated_states_rates = []
            for line in lines:
                if 'error_rate' in line:
                    er = line.split(' ')[1].replace('%', '')
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
                    d_usr = line.split(' ')[0].split(':')[1]
                    d_updated_states_rates.append(float(d_usr) * 100)
            cps = getCPS(skipped_layers, [v_cps, a_cps, d_cps], True)
            # print(cps)
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
                data[skipped_layers][cps].append([train_error_rates[-1], 'Train Error Rate'])
                data[skipped_layers][cps].append([error_rates[-1], 'Error Rate'])
                data[skipped_layers][cps].append([d_updated_states_rates[-1], 'Decoder Update Rate'])
    data_combi = {key:value for key,value in data.items() if len(key) > 1}
    data_single = {key:value for key,value in data.items() if len(key) == 1}

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
                    mean_values.append([skipped_layer+':'+cps_values[skipped_layer], skipped_layer_value[0], skipped_layer_value[1]])
            df = pd.DataFrame(value, columns=['cost_per_sample', 'value', 'Metric'])
            mean_df = pd.DataFrame(mean_values, columns=['cost_per_sample', 'value', 'Metric'])
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
                plt.axhline(y=line_data[('d:' + cps_values['d'], 'Error Rate')], color=colors[3], linestyle='dotted', label='(old) Decoder Update Rate', linewidth=1.5)
                plt.axhline(y=line_data[('d:' + cps_values['d'], 'Decoder Update Rate')], color=colors[3], label='(old) Decoder Error Rate', linewidth=1.5)

            print(df)
            box_plot = sns.boxplot(y='value', x='cost_per_sample', data=df, palette='bright', hue='Metric',
                                   ax=ax, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"5"},
                                   boxprops=dict(alpha=.3))
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
            plt.title(title)
            plt.xlabel('Cost per Sample')
            plt.ylabel('Percentage')
            plt.show()
            #cps_fn = '_'.join(cps_values.values()).replace('.','')
            #plt.savefig('./visualization/boxplot_compareLines_' + key + '_' + cps_fn + '.png', dpi=320)

#visualizeCombinedExperiments()
#visualizeCombinedExperiments2()
visualizeExperiment(architecture=['bimodal'], save=True)