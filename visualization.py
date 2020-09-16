import numpy as np
import matplotlib.pyplot as plt

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
                    
visualize_multiple_exp()