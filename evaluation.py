import os
import json
import numpy as np
import seaborn as sns

def rewrite_log():
    log_dir = "./logs/"
    file_name = "grid_av_to_chars_501__0001_10_tanh_False_10_test_train_error_rate"
    dict = {}
    with open(log_dir + file_name, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            if 'character' in l:
                l = l.rsplit("% ")[:-1]
                for pair in l:
                    name, value = pair.split(": ")
                    value = float(value)
                    if name in dict.keys():
                        dict[name].append(value)
                    else:
                        dict[name] = [value]
    reshaped_values = []
    for i in range(len(list(dict.values())[0])):
        x = [str(dict[key][i]) for key in dict.keys()]
        reshaped_values.append(x)
    with open(log_dir + file_name + "_rewritten.csv", 'w') as f:
        f.write('\t'.join(dict.keys()))
        f.write('\n')
        for x in reshaped_values:
            f.write('\t'.join(x))
            f.write('\n')
            
def get_data_ratio():
    dataset_dir = 'F:/Documents/PycharmProjects/Masterthesis/datasets/mvlrs_v1/'
    ratio = {}
    all = 0
    for file in [f for f in os.listdir(dataset_dir) if '.txt' in f]:
        ratio[file] = 0
        with open(dataset_dir+file, 'r') as f:
            for line in f.read().splitlines():
                ratio[file] += 1
                all += 1
    print(ratio)
    for v in ratio.values():
        print(v/all)
        
def create_config_files():
    dataset_dir = 'F:/Documents/datasets/mvlrs_v1/'
    training = []
    testing = []
    for file in [f for f in os.listdir(dataset_dir) if '.txt' in f and not '_' in f]:
        with open(dataset_dir + file, 'r') as f:
            for line in f.read().splitlines():
                if 'train' in file:
                    if file == 'pretrain.txt':
                        training.append("pretrain/"+line+".mp4\n")
                    else:
                        training.append("main/" + line + ".mp4\n")
                else:
                    if file == 'test.txt':
                        line = line.split(' ')[0] + '.mp4\n'
                        testing.append("main/"+line)
                    else:
                        testing.append("main/" + line + ".mp4\n")
    with open(dataset_dir+"pretrain_train.txt", 'w') as f:
        f.writelines(training)
    with open(dataset_dir+"val_test.txt", 'w') as f:
        f.writelines(testing)

def create_config_files2():
    dataset_dir = 'F:/Documents/datasets/mvlrs_v1/'
    files = []
    data = 'pretrain.txt'
    prefix = 'pretrain/' if data == 'pretrain.txt' else 'main/'
    with open(dataset_dir+data) as f:
        lines = f.read().splitlines()
        for line in lines:
            if data == 'test.txt':
                line = line.split(' ')[0]
            files.append(prefix + line + '.mp4\n')
    with open(dataset_dir+'video_files_'+data, 'w') as f:
        f.writelines(files)

import pickle as p
import matplotlib.pyplot as plt
def analyze_alignment():
    eval_data = p.load(open('./eval_data/eval_data_e25.p', 'rb'))
    for key, val in eval_data.items():
        print(key)
        img = val['img']
        av = val['av']
        updated_states = val['updated_states']
        #plt.imshow(img)
        #plt.show()
        #plt.imshow(av)
        #plt.show()
        plt.imshow(updated_states.reshape(updated_states.shape[0], updated_states.shape[1]))
        plt.title(key)
        plt.show()

from PIL import Image
import numpy as np
def analyze_alignment_img():
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    for epoch in range(1,70):
        eval_data = p.load(open(f'./eval_data/mvlrs_v1/exp1_av_skip/eval_data_e{epoch}.p', 'rb'))
        res_same_attention = [0, 0]
        res_state_change = [0, 0, 0]
        avg_attention = []
        max_attention = 0
        min_attention = 1
        small_diff = 0
        big_diff = 0
        for key in eval_data.keys():
            updated_states = eval_data[key]['audio_updated_states'][:audio_input_length[key]]
            data = eval_data[key]['av'][:audio_input_length[key]]
            if data.shape[1] == 30:
                data = data.transpose()
            for row in data:
                row = row[:audio_input_length[key]]
                #norm = np.sum(row)
                norm = 1.0
                for i, col in enumerate(row):
                    if updated_states[i] == 0:
                        if norm > 0.0:
                            avg_attention.append(col[0]/norm)
                            if col[0]/norm < min_attention:
                                min_attention = col[0]/norm
                            if col[0]/norm > max_attention:
                                max_attention = col[0]/norm
                    if updated_states[i] == 0 and updated_states[i-1] == 0: #same_attention skipped_states
                        if row[i-1][0] == col[0]:
                            res_same_attention[0] = res_same_attention[0] + 1
                        else:
                            res_same_attention[1] = res_same_attention[1] + 1
                    if updated_states[i] == 0 and updated_states[i-1] == 1:
                        if row[i-1][0] < col[0]:
                            small_diff += col[0]-row[i-1][0]
                            res_state_change[0] = res_state_change[0] + 1
                        if row[i-1][0] == col[0]:
                            res_state_change[1] = res_state_change[1] + 1
                        if row[i-1][0] > col[0]:
                            big_diff += row[i - 1][0] - col[0]
                            res_state_change[2] = res_state_change[2] + 1
        print(min_attention, np.mean(avg_attention), max_attention)
        print(res_same_attention, res_same_attention[0]/sum(res_same_attention))
        print(res_state_change, res_state_change[0]/sum(res_state_change), res_state_change[1]/sum(res_state_change), res_state_change[2]/sum(res_state_change))
        #print(small_diff/res_state_change[0])
        #print(big_diff/res_state_change[2])
    # Res: alle skipped_states haben die gleichen Werte
    # updated_state -> skipped_state haben unterschiedliche attention
    #
    
def compare_alignment(eval_data_1=None, eval_data_2=None):
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    if not eval_data_1 and not eval_data_2:
        eval_data_1 = p.load(open(f'./eval_data/mvlrs_v1/exp20_av_skip_comb_drop/eval_data_e{63}.p', 'rb'))
        eval_data_2 = p.load(open(f'./eval_data/mvlrs_v1/exp18_normal_drop/eval_data_e{56}.p', 'rb'))
    else:
        print(eval_data_1)
        print(eval_data_2)
        eval_data_1 = p.load(open(eval_data_1, 'rb'))
        eval_data_2 = p.load(open(eval_data_2, 'rb'))
    greater = 0
    equal = 0
    lower = 0
    total = 0
    min_att = 1
    max_att = 0
    greater_diff = []
    lower_diff = []
    for key in eval_data_1.keys():
        updated_states = eval_data_1[key]['audio_updated_states'][:audio_input_length[key]]
        data_1 = eval_data_1[key]['av']
        data_1 = data_1[:video_input_length[key]]
        data_2 = eval_data_2[key]['av']
        data_2 = data_2[:video_input_length[key]]
        for r,row in enumerate(data_1):
            row = row[:audio_input_length[key]]
            #norm_1 = np.sum(row)
            #norm_2 = np.sum(data_2[r][:audio_input_length[key]])
            norm_1, norm_2 = 1.0, 1.0
            #print(norm_1, norm_2)
            for c, col in enumerate(row):
                if updated_states[c] == 1:# and updated_states[c-1] == 1:
                    if norm_1 > 0 and norm_2 > 0:
                        v_1 = col/norm_1
                        v_2 = data_2[r][c]/norm_2
                        if v_1 < min_att: min_att = v_1
                        if v_1 > max_att: max_att = v_1
                        if v_1 > v_2:
                            greater += 1
                            greater_diff.append(v_1-v_2)
                        elif v_1 == v_2: equal += 1
                        elif v_1 < v_2:
                            lower += 1
                            lower_diff.append(v_2-v_1)
                        else: print(v_1, v_2)
                        total += 1
    print('greater', greater, greater/total)
    print('equal', equal, equal/total)
    print('lower', lower, lower/total)
    print('min_att', min_att)
    print('max_att', max_att)
    print(np.mean(greater_diff), np.max(greater_diff), np.min(greater_diff))
    print(np.mean(lower_diff), np.max(lower_diff), np.min(lower_diff))
    
def compare_multiple_alignments():
    import glob
    for n_folder_nr in [1,2,3,4,5]:
        n_list_of_files = glob.glob(f'./eval_data/mvlrs_v1/exp{n_folder_nr}_attention_test/*')
        n_latest_file = max(n_list_of_files, key=os.path.getctime)
        for s_folder_nr in [1,2,3,4,5]:
            s_list_of_files = glob.glob(f'./eval_data/mvlrs_v1/exp{s_folder_nr}_a_skip_attention_test/*')
            s_latest_file = max(s_list_of_files, key=os.path.getctime)
            compare_alignment(s_latest_file, n_latest_file)
        
#compare_multiple_alignments()
#analyze_alignment_img()

def test323():
    import glob
    s_list_of_files = glob.glob(f'./eval_data/mvlrs_v1/exp{1}_a_skip_attention_test/*')
    s_latest_file = max(s_list_of_files, key=os.path.getctime)
    eval_data_1 = p.load(open(s_latest_file, 'rb'))
    for key in eval_data_1.keys():
        print(key)
        updated_states = eval_data_1[key]['audio_updated_states']
        plt.plot(updated_states)
        plt.show()

#import tensorflow as tf

#for example in tf.python_io.tf_record_iterator("N:/datasets/mvlrs_v1/tfrecords/logmel_test_clean.tfrecord"):
#    print(tf.train.Example.FromString(example))

#test323()

#eval_data = p.load((open('./eval_data/mvlrs_v1/exp1_av_skip/eval_data_e1.p','rb')))
#for key in eval_data.keys():
#    video_input_length = p.load(open('./datasets/' + 'mvlrs_v1' + '/audio_seq_len.p', 'rb'))
#    print(eval_data[key]['audio_updated_states'].shape)
#    print(video_input_length[key])

#analyze_alignment_img()

#compare_alignment()

# updated_states = np.array([[[1.], [1.], [1.], [1.], [0.]],
#                            [[1.], [1.], [1.], [1.], [0.]],
#                            [[1.], [1.], [1.], [1.], [0.]],
#                            [[1.], [1.], [1.], [1.], [0.]],])
#
# output = np.array([[[1.1, 2.1, 3.1, 4.1, 5.1], [1.2, 2.2, 3.2, 4.2, 5.2], [1.3, 2.3, 3.3, 4.3, 5.3], [1.4, 2.4, 3.4, 4.4, 5.4], [1.5, 2.5, 3.5, 4.5, 5.5]],
#                    [[1.1, 2.1, 3.1, 4.1, 5.1], [1.2, 2.2, 3.2, 4.2, 5.2], [1.3, 2.3, 3.3, 4.3, 5.3], [1.4, 2.4, 3.4, 4.4, 5.4], [1.5, 2.5, 3.5, 4.5, 5.5]],
#                    [[1.1, 2.1, 3.1, 4.1, 5.1], [1.2, 2.2, 3.2, 4.2, 5.2], [1.3, 2.3, 3.3, 4.3, 5.3], [1.4, 2.4, 3.4, 4.4, 5.4], [1.5, 2.5, 3.5, 4.5, 5.5]],
#                    [[1.1, 2.1, 3.1, 4.1, 5.1], [1.2, 2.2, 3.2, 4.2, 5.2], [1.3, 2.3, 3.3, 4.3, 5.3], [1.4, 2.4, 3.4, 4.4, 5.4], [1.5, 2.5, 3.5, 4.5, 5.5]],])
#
# batch_size = 4
# print(updated_states.shape)
# print(output.shape)
# new_h = output[np.where(updated_states.reshape((updated_states.shape[0],updated_states.shape[1])) == 1.)]
# print(new_h.shape)
# print(new_h.reshape((batch_size,new_h.shape[0]/batch_size,new_h.shape[1])))

def plotUpdatesStates():
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    import glob
    for n_folder_nr in [1]:
        video_count = 0
        audio_count = 0
        video_max_length = max([video_input_length[key] for key in video_input_length.keys()])
        audio_max_length = max([audio_input_length[key] for key in audio_input_length.keys()])
        print('video_max_length', video_max_length)
        print('audio_max_length', audio_max_length)
        video_img = []
        audio_img = []
        n_list_of_files = glob.glob(f'./eval_data/mvlrs_v1/exp{n_folder_nr}_av_skip/*')
        n_latest_file = max(n_list_of_files, key=os.path.getctime)
        eval_data_1 = p.load(open(n_latest_file, 'rb'))
        for key in eval_data_1.keys():
            video_updated_states = eval_data_1[key]['video_updated_states'][:video_input_length[key]]
            audio_updated_states = eval_data_1[key]['audio_updated_states'][:audio_input_length[key]]
            video_s = [x[0] for x in eval_data_1[key]['video_updated_states']]
            audio_s = [x[0] for x in eval_data_1[key]['audio_updated_states']]
            for _ in range(video_max_length-video_input_length[key]):
                video_s.append(0.5)
            for _ in range(audio_max_length-audio_input_length[key]):
                audio_s.append(0.5)
            video_img.append(video_s)
            audio_img.append(audio_s)
            for x in eval_data_1[key]['video_updated_states']:
                if x == 0: video_count += 1
            for x in eval_data_1[key]['audio_updated_states']:
                if x == 0: audio_count += 1
        print(video_count)
        video_img = np.array(video_img)
        plt.imshow(np.rot90(video_img))
        plt.show()
        print(audio_count)
        audio_img = np.array(audio_img)
        plt.imshow(np.rot90(audio_img))
        plt.show()
  
def boxplotAttentionValues():
    import glob
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    attention_values_normal = []
    attention_values_skip = []
    for n_folder_nr in [1,2,3,4,5]:
        n_list_of_files = glob.glob(f'./eval_data/mvlrs_v1/exp{n_folder_nr}_attention_test/*')
        n_latest_file = max(n_list_of_files, key=os.path.getctime)
        eval_data_1 = p.load(open(n_latest_file, 'rb'))
        for ki, key in enumerate(eval_data_1.keys()):
            if ki % 100 == 0: print(ki, len(eval_data_1.keys()))
            data_1 = eval_data_1[key]['av']
            data_1 = data_1[:video_input_length[key]]
            for r, row in enumerate(data_1):
                row = row[:audio_input_length[key]]
                norm_1 = np.sum(row)
                for c, col in enumerate(row):
                    v_1 = col / norm_1
                    attention_values_normal.append(v_1[0])
    for s_folder_nr in [1,2,3,4,5]:
        s_list_of_files = glob.glob(f'./eval_data/mvlrs_v1/exp{s_folder_nr}_av_skip/*')
        s_latest_file = max(s_list_of_files, key=os.path.getctime)
        eval_data_1 = p.load(open(s_latest_file, 'rb'))
        for ki, key in enumerate(eval_data_1.keys()):
            if ki % 100 == 0: print(ki, len(eval_data_1.keys()))
            data_1 = eval_data_1[key]['av']
            data_1 = data_1[:video_input_length[key]]
            for r, row in enumerate(data_1):
                row = row[:audio_input_length[key]]
                norm_1 = np.sum(row)
                for c, col in enumerate(row):
                    v_1 = col / norm_1
                    attention_values_skip.append(v_1[0])
    data = [attention_values_normal, attention_values_skip]
    fig, ax = plt.subplots()
    ax.boxplot(data)
    plt.show()
    
def test13():
    import glob
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    n_list_of_files = glob.glob(f'./eval_data/mvlrs_v1/av_align/vad/00/131/exp0/mvlrs_v1_av_align_vad_00_131_exp0/*')
    n_latest_file = max(n_list_of_files, key=os.path.getctime)
    eval_data_1 = p.load(open(n_latest_file, 'rb'))
    for key in eval_data_1.keys():
        data = eval_data_1[key]['decoder_updated_states'][:audio_input_length[key]]
        print(data.shape)
        print(video_input_length[key])
        print(audio_input_length[key])
        print(np.sum(data, (0)))
        plt.imshow(data)

def minUpdateRate():
    train_file = open('./datasets/mvlrs_v1/train.scp', 'r')
    train_file_names = train_file.read().splitlines()
    print(len(train_file_names))

    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    a_update_rate = 0
    v_update_rate = 0
    for key in train_file_names:
        a_update_rate += 1/audio_input_length[key]
        v_update_rate += 1/video_input_length[key]
    print('min_audio_update_rate', a_update_rate/len(audio_input_length.keys()))
    print('min_video_update_rate', v_update_rate/len(video_input_length.keys()))

def formatCPS(cps):
    return format(eval(cps), 'f').rstrip('0').rstrip('.')

def getAllLogs(logsPath):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(logsPath):
        if len(filenames) == 1:
            f.append([dirpath, filenames[0]])
    return f

def getParamsOfLog(logDir):
    params = logDir.split('/')[4:]
    return params

def timeOfExperiments():
    f = getAllLogs('./logs/mvlrs_v1/combined experiment/av_align/vad/')
    times = []
    for logDir, logName in f:
        architecture, skipped_layers, v_cps, a_cps, d_cps, number_of_layers, exp_number = getParamsOfLog(logDir)
        v_cps = formatCPS(v_cps)
        a_cps = formatCPS(a_cps)
        d_cps = formatCPS(d_cps)
        with open(logDir+'/'+logName, 'r') as f:
            lines = f.read().splitlines()
            times.append(float(lines[-1].split(':')[1]) - float(lines[0].split(':')[1]))
    print(np.mean(times)/60/60)
    print(np.min(times)/60/60)
    print(np.max(times)/60/60)

#timeOfExperiments()

def getImportantTokens():
    configs_path = './configs/mvlrs_v1/noise_configs/'
    combined_audio_updated_states = {}
    for config_file in os.listdir(configs_path):
        config = json.load(open(configs_path + config_file, 'r'))
        experiment_path = config['experiment_path']
        experiment_name = config['experiment_name']
        combined_experiment_name = experiment_name[:-5]
        if combined_experiment_name not in combined_audio_updated_states.keys():
            combined_audio_updated_states[combined_experiment_name] = {}
        audio_updated_states = p.load(open('./eval_data/' + experiment_path + experiment_name + '/audio_updated_states.p', 'rb'))
        for key, value in audio_updated_states.items():
            if key not in combined_audio_updated_states[combined_experiment_name].keys():
                combined_audio_updated_states[combined_experiment_name][key] = value
            else:
                combined_audio_updated_states[combined_experiment_name][key] = np.add(combined_audio_updated_states[combined_experiment_name][key], value)
    p.dump(combined_audio_updated_states, open('./eval_data/combined_audio_updated_states.p', 'wb'))
    
#getImportantTokens()

def selectTokens():
    combined_audio_updated_states = p.load(open('./eval_data/combined_audio_updated_states.p', 'rb'))
    zero_audio_tokens = {}
    for combined_experiment_name, audio_updated_states in combined_audio_updated_states.items():
        zero_audio_tokens[combined_experiment_name] = {}
        print(combined_experiment_name)
        for audio_file, value in audio_updated_states.items():
            value = value.reshape(1, len(value))[0]
            indices = np.where(value > 1)[0]
            i = int(np.floor(len(indices)*0.1))
            x = np.random.choice(indices, i)
            zero_audio_tokens[combined_experiment_name][audio_file] = sorted(x)
    p.dump(zero_audio_tokens, open('./eval_data/zero_audio_tokens.p', 'wb'))
    
#selectTokens()

def combined_selectedTokens():
    combined_audio_updated_states = p.load(open('./eval_data/combined_audio_updated_states.p', 'rb'))
    percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    zero_audio_tokens = {}
    combined_zero_audio_tokens = {perc:{} for perc in percentages}
    for audio_file, updated_states in combined_audio_updated_states[list(combined_audio_updated_states.keys())[0]].items():
        for key in combined_audio_updated_states.keys():
            if audio_file in zero_audio_tokens.keys():
                zero_audio_tokens[audio_file] = np.add(zero_audio_tokens[audio_file], combined_audio_updated_states[key][audio_file])
            else:
                zero_audio_tokens[audio_file] = combined_audio_updated_states[key][audio_file]
        #print(zero_audio_tokens[audio_file].reshape(1, len(zero_audio_tokens[audio_file]))[0])
        indices = np.where(zero_audio_tokens[audio_file].reshape(1, len(zero_audio_tokens[audio_file]))[0] >= 14)[0]
        for perc in percentages:
            i = int(np.ceil(len(indices) * perc))
            x = np.random.choice(indices, i)
            combined_zero_audio_tokens[perc][audio_file] = x
    #print(np.min(lengths), np.mean(lengths), np.max(lengths))
    p.dump(combined_zero_audio_tokens, open('./eval_data/combined_zero_audio_tokens.p', 'wb'))

#combined_selectedTokens()
#a[zeroing] = 0

#combined_zero_audio_tokens = p.load(open('F:\Documents\PycharmProjects\Masterthesis\skip-avsr\eval_data\combined_zero_audio_tokens.p', 'rb'))
#audio_seq_len = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
#print(list(combined_zero_audio_tokens[0.1]['main/5535415699068794046/00002']))
#print(audio_seq_len['main/5535415699068794046/00002'])
#f = open('F:/Documents/datasets/mvlrs_v1/splits/train.scp', 'r')
#for line in f.read().splitlines():
#    #print(combined_zero_audio_tokens[0.1].keys())
#    if not line in combined_zero_audio_tokens[0.1]:
#        print('error', line)

#encoder_attention_summary, encoder_attention_alignment, decoder_attention_summary, decder_attention_alignment, video_updated_states, audio_updated_states, decoder_updated_states
def show_mean_attention_updated_states():
    overall_mean = []
    path = './configs/mvlrs_v1/evaluate/finished/'
    for file in os.listdir(path):
        if 'av_align' in file and 'clean' in file and 'vad' in file:
            config = json.load(open(path+file, 'r'))
            experiment_path = config['experiment_path']
            experiment_name = config['experiment_name']
            a = p.load(open('./eval_data/'+experiment_path+experiment_name+'/eval_data_T_24_e1111.p', 'rb'))
            mean_attention_skipped_states = []
            for key in a.keys():
                eas = a[key]['encoder_attention_alignment']
                vus = a[key]['video_updated_states']
                aus = a[key]['audio_updated_states']
                eas = eas[:vus.shape[0],:aus.shape[0]].reshape((vus.shape[0], aus.shape[0]))
                va_us = np.add(np.tile(vus, aus.shape[0]), np.tile(aus, vus.shape[0]).T)
                va_us = np.where(va_us==0, 1, 0)
                ones = eas[np.where(va_us == 1)]
                for x in ones:
                    mean_attention_skipped_states.append(x)
                #if config['cost_per_sample'] == [0.0001, 0.0005, 0.0001]:
                    # plt.boxplot(mean_attention_skipped_states)
                #    plt.imshow(eas)
                #    plt.show()
                #if len(ones) > 0:
                #    mean_attention_skipped_states.append(np.mean(ones))
            #print(len(mean_attention_skipped_states))
            if np.mean(mean_attention_skipped_states) > 0:
                print(np.mean(mean_attention_skipped_states))
                overall_mean.append(np.mean(mean_attention_skipped_states))
    print(np.mean(overall_mean))

def compare_udpated_states():
    path = './configs/evaluate/finished/'
    dataset = 'mvlrs_v1'
    noise = 'clean'
    architecture = 'av_align'
    skip_layer = 'a'
    for cps in [['0E+00','0E+00','0E+00'],
                ['0E+00', '1E-02', '0E+00'],
                ['0E+00', '1E-03', '0E+00'],
                ['0E+00', '1E-04', '0E+00'],
                ['0E+00', '1E-05', '0E+00'],
                ['0E+00', '5E-04', '0E+00']]:
        similarity = []
        identical_us = []
        xs = {}
        ys = {}
        for exp in ['0','1','2']:
            a_config_file = dataset+'_'+noise+'_'+architecture+'_'+skip_layer+'_'+'_'.join(cps)+'_131_exp'+exp+'.json'
            a_config = json.load(open(path + a_config_file, 'r'))
            a_experiment_path = a_config['experiment_path']
            a_experiment_name = a_config['experiment_name']
            a = p.load(open('./eval_data/' + a_experiment_path + a_experiment_name + '/eval_data_T_24_e1111.p', 'rb'))
            d_config_file = dataset + '_' + noise + '_' + architecture + '_' + 'd' + '_' + '_'.join(
                [cps[0],cps[2],cps[1]]) + '_131_exp' + exp + '.json'
            d_config = json.load(open(path + d_config_file, 'r'))
            d_experiment_path = d_config['experiment_path']
            d_experiment_name = d_config['experiment_name']
            d = p.load(open('./eval_data/' + d_experiment_path + d_experiment_name + '/eval_data_T_24_e1111.p', 'rb'))
            for key in a.keys():
                if 'audio_updated_states' in a[key].keys():
                    x = [n[0] for n in a[key]['audio_updated_states']]
                    y = [n[0] for n in d[key]['decoder_updated_states']]
                    if not key in xs.keys():
                        xs[key] = x
                        ys[key] = y
                    else:
                        xs[key] = np.add(xs[key], x)
                        ys[key] = np.add(ys[key], y)
        for key in xs.keys():
            cos_sim = np.dot(xs[key], ys[key])/(np.linalg.norm(xs[key])*np.linalg.norm(ys[key]))
            similarity.append(cos_sim)
            identical = 0
            for i in range(len(x)):
                if x[i] == y[i]:
                    identical += 1
                identical_us.append(identical / len(x))
        print(cps, np.mean(similarity))
        print(cps, np.mean(identical_us))
    
#compare_udpated_states()

def identify_updates_states():
    path = './configs/evaluate/finished/'
    for file in os.listdir(path):
        if 'exp2' in file:
            print(file)
            v_states = [0] * 100
            a_states = [0] * 100
            d_states = [0] * 100
            total_v = [0] * 100
            total_a = [0] * 100
            total_d = [0] * 100
            if 'av_align' in file and 'clean' in file:
                config = json.load(open(path + file, 'r'))
                experiment_path = config['experiment_path']
                experiment_name = config['experiment_name']
                a = p.load(open('./eval_data/' + experiment_path + experiment_name + '/eval_data_T_24_e1111.p', 'rb'))
                for key in a.keys():
                    if 'video_updated_states' in a[key].keys():
                        vus = a[key]['video_updated_states']
                        for i, u in enumerate(vus):
                            x = int(np.round(i/len(vus)*100))
                            v_states[x] += u[0]
                            total_v[x] += 1
                    elif 'audio_updated_states' in a[key].keys():
                        aus = a[key]['audio_updated_states']
                        for i, u in enumerate(aus):
                            x = int(np.round(i/len(aus)*100))
                            a_states[x] += u[0]
                            total_a[x] += 1
                    elif 'decoder_updated_states' in a[key].keys():
                        dus = a[key]['decoder_updated_states']
                        for i, u in enumerate(dus):
                            x = int(np.ceil(i/len(dus)*100))
                            d_states[x] += u[0]
                            total_d[x] += 1
            print('v', v_states)
            print('a', a_states)
            print('d', d_states)
            print(len(a_states))
            if sum(a_states) > 0:
                plt.bar([x for x in range(0,100)], [a_states[i]/total_a[i] for i in range(0,100)])
                plt.title(experiment_name)
                plt.show()
            
def analyze_bimodal_decoder_attention():
    '''
    Possible influence visible in the decoder attention for audio and video.
    What I saw was the usual attention image, similar to the machine translation task for text, but only
    for the audio channel and not the video channel.
    '''
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    label_length = p.load(open('./datasets/mvlrs_v1/label_length.p', 'rb'))
    for config_file in os.listdir('./configs/analyze/bimodal_decoder/'):
        config = json.load(open('./configs/analyze/bimodal_decoder/' + config_file, 'r'))
        experiment_path = config['experiment_path']
        experiment_name = config['experiment_name']
        dataset, noise, arch, skip_layer, vcps, acps, dvcps, dacps, _, _, _ = experiment_path.split('/')
        x = p.load(open('./eval_data/' + experiment_path + experiment_name + '/eval_data_T_24_e1112.p', 'rb'))
        file = 'main/5681963419182164492/00029'
        #file = list(x.keys())[np.random.randint(0,len(x.keys()))]
        print(file)
        daav = x[file]['decoder_attention_alignment_video'][:video_input_length[file], :label_length[file]['length']]
        daav = np.subtract(np.ones_like(daav), daav)
        for i in range(len(daav)):
            daav[i] = daav[i] * 1/sum(daav[i])
        print(sum(daav[0]))
        daaa = x[file]['decoder_attention_alignment_audio'][:audio_input_length[file], :label_length[file]['length']]
        daaa = np.subtract(np.ones_like(daaa), daaa)
        for i in range(len(daaa)):
            daaa[i] = daaa[i] * 1/sum(daaa[i])
        print(sum(daaa[0]))
        print(skip_layer)
        s = 2
        if 'd' in skip_layer:
            s += 2
            
        fig, ax = plt.subplots(1,s)
        text = [c for c in label_length[file]['text']]
        text_arr = np.arange(len(text))
        print(text)
        if 'd' in skip_layer:
            ax[0].imshow(x[file]['decoder_updated_states_video'][:video_input_length[file]])
            ax[0].set_xticklabels([])
            ax[0].tick_params(axis=u'x', which=u'both', length=0)
            ax[0].set_ylabel('Video Token')
            ax[1].imshow(daav, aspect="auto")
            ax[1].set_xticks(text_arr)
            ax[1].set_xticklabels(text)
            ax[1].set_yticklabels([])
            ax[1].tick_params(axis=u'y', which=u'both', length=0)
            ax[2].imshow(x[file]['decoder_updated_states_audio'][:audio_input_length[file]])
            ax[2].set_xticklabels([])
            ax[2].tick_params(axis=u'x', which=u'both', length=0)
            ax[2].set_ylabel('Audio Token')
            ax[3].imshow(daaa, aspect="auto")
            ax[3].set_xticks(text_arr)
            ax[3].set_xticklabels(text)
            ax[3].set_yticklabels([])
            ax[3].tick_params(axis=u'y', which=u'both', length=0)
        else:
            ax[0].imshow(daav, aspect="auto")
            ax[0].set_xticks(text_arr)
            ax[0].set_xticklabels(text)
            ax[0].set_ylabel('Video Token')
            ax[1].imshow(daaa, aspect="auto")
            ax[1].set_xticks(text_arr)
            ax[1].set_xticklabels(text)
            ax[1].set_ylabel('Audio Token')
        title = ' '.join([dataset.upper(), noise.upper(), skip_layer.upper(), vcps, acps, dvcps, dacps])
        fig.suptitle(title)
        plt.show()


def analyze_decoder_attention():
    '''
    Possible influence visible in the decoder attention for audio and video.
    What I saw was the usual attention image, similar to the machine translation task for text, but only
    for the audio channel and not the video channel.
    '''
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    label_length = p.load(open('./datasets/mvlrs_v1/label_length.p', 'rb'))
    files = None
    file = None
    text = "0" * 100
    for config_file in os.listdir('./configs/analyze/decoder_attention/'):
        print(config_file)
        for o_file in range(100):
            print(o_file)
            c1 = config_file.replace('cafe_10db','clean')
            c2 = c1
            c3 = config_file
            for i,j in [('vad','n'),('1E','0E'), ('5E','0E'),('-','+'),('4','0'),('03','00'),('5','0')]:
                c2 = c2.replace(i, j)
                c3 = c3.replace(i, j)
            cfiles = [c2, c3, c1, config_file]
            s = 0
            fig, ax = plt.subplots(2, 4)
            d = {'clean':None, 'cafe_10db':None}
            for c_file in cfiles:
                config = json.load(open('./configs/finished/' + c_file, 'r'))
                experiment_path = config['experiment_path']
                experiment_name = config['experiment_name']
                dataset, noise, arch, skip_layer, vcps, acps, dcps, _, _, _ = experiment_path.split('/')
                x = p.load(open('./eval_data/' + experiment_path + experiment_name + '/eval_data_T_24_e1112.p', 'rb'))
                if not files:
                    files = []
                    while len(files) < 100:
                        files.append(list(x.keys())[np.random.randint(0, len(x.keys()))])
                        files = list(set(files))
                #file = files[o_file]
                #text = [c for c in label_length[file]['text']]
                    #while len(text) > 20:
                    #    file = list(x.keys())[np.random.randint(0, len(x.keys()))]
                    #file = 'main/5681963419182164492/00029'
                file = files[o_file]
                text = [c for c in label_length[file]['text']]
                text_arr = np.arange(len(text))
                daa = x[file]['decoder_attention_alignment'][:video_input_length[file], :label_length[file]['length']]
                eaa = x[file]['encoder_attention_alignment'][:video_input_length[file], :audio_input_length[file]]
                eaa_max = str(np.round(np.max(eaa), 4))
                if 'vad' == skip_layer:
                    eaa = np.concatenate((eaa, x[file]['video_updated_states'][:, None]), axis=1)
                    t = x[file]['audio_updated_states']
                    t = [np.vstack((t, [0.5]))]
                    eaa = np.vstack((eaa,t))
                #print(eaa.shape)
                #daa = np.subtract(np.ones_like(daa), daa)
                #for i in range(len(daa)):
                #    daa[i] = daa[i] * 1 / sum(daa[i])
                #print('sum', sum(daa[:,0]))
                #print('max daa', np.max(daa))
                #print('max eaa', np.max(eaa))
                im = ax[1][s].imshow(daa, aspect="auto", vmin=0, vmax=1)
                ax[1][s].set_xticks(text_arr)
                ax[1][s].set_xticklabels(text)
                ax[1][s].set_ylabel('A/V Token')
                im = ax[0][s].imshow(eaa, aspect="auto", vmin=0, vmax=1)
                ax[0][s].set_xlabel('Audio Token'+'\n'+eaa_max)
                ax[0][s].set_ylabel('Video Token')
                if 'vad' == skip_layer:
                    image_title = noise+' '+skip_layer+' '+vcps+' '+acps+' '+dcps
                    ax[0][s].set_title(image_title.upper())
                else:
                    image_title = noise + ' ' + skip_layer
                    ax[0][s].set_title(image_title.upper())
                s += 1
            title = ' '.join([dataset.upper(), arch, vcps, acps, dcps])
            fig.suptitle(title)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            #plt.show()
            os.makedirs('./visualization/attention_alignment/'+title+'/', exist_ok=True)
            fig.set_size_inches(18.5, 10.5)
            plt.savefig('./visualization/attention_alignment/'+title+'/'+str(o_file)+'.png', dpi=320)
            plt.close()

def analyzePredictions():
    predictions = p.load(open('./predictions/predictions.p','rb'))
    file = None
    for config_file in os.listdir('./configs/analyze/decoder_attention/'):
        c1 = config_file.replace('cafe_10db','clean')
        if not file:
            c2 = c1
            c3 = config_file
            for i, j in [('vad', 'n'), ('1E', '0E'), ('5E', '0E'), ('-', '+'), ('4', '0'), ('03', '00'), ('5', '0')]:
                c2 = c2.replace(i, j)
                c3 = c3.replace(i, j)
            cfiles = [c2, c3, c1, config_file]
        else:
            cfiles = [c1, config_file]
        for c_file in cfiles:
            config = json.load(open('./configs/finished/' + c_file, 'r'))
            experiment_name = config['experiment_name']
            experiment_predictions = predictions[experiment_name]
            if not file:
                file = list(experiment_predictions.keys())[np.random.randint(0, len(experiment_predictions.keys()))]
                print(file)
                file_prediction = experiment_predictions[file]
                print(file_prediction['target'])
            else:
                file_prediction = experiment_predictions[file]
            print('{:60s} {:1.4f}  {:10s}'.format(experiment_name, file_prediction['error'], file_prediction['pred']))
        print()
        
def analyze_attention_values(location='encoder'):
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    label_length = p.load(open('./datasets/mvlrs_v1/label_length.p', 'rb'))
    res = {}
    files = None
    for config_file in os.listdir('./configs/analyze/attention/'):
        config = json.load(open('./configs/finished/' + config_file, 'r'))
        experiment_path = config['experiment_path']
        experiment_name = config['experiment_name']
        print(experiment_name)
        if config['architecture'] == 'av_align' and config['dataset'] == 'mvlrs_v1' and ('clean' in config['snr']):#' '.join([dataset.upper(), arch, skip_layer, vcps, acps, dcps])
            dataset, noise, arch, skip_layer, vcps, acps, dcps, _, _, _ = experiment_path.split('/')
            x = p.load(open('./eval_data/' + experiment_path + experiment_name + '/eval_data_T_24_e1112.p', 'rb'))
            eaa_max_list = []
            eaa_mean_max_list = []
            count = 0
            for file in x.keys():
                if file != 'flops':
                    if location == 'decoder':
                        eaa = x[file]['decoder_attention_alignment'][:audio_input_length[file],
                              :label_length[file]['length']]
                    else:
                        eaa = x[file]['encoder_attention_alignment'][:video_input_length[file], :audio_input_length[file]]
                    if count < 25:
                        fig = plt.figure()
                        if not files:
                            files = []
                            while len(files) < 25:
                                f = list(x.keys())[np.random.randint(0,len(x.keys()))]
                                files.append(f)
                                files = list(set(files))
                        print(count)
                        if not 'n' in skip_layer:
                            title = ' '.join([dataset.upper(), arch, skip_layer, 'V'+vcps, 'A'+acps, 'D'+dcps])
                        else:
                            title = ' '.join([dataset.upper(), arch, skip_layer])
                        if location == 'decoder':
                            img = x[files[count]][location+'_attention_alignment'][:video_input_length[files[count]],
                              :label_length[file]['length']]
                        else:
                            img = x[files[count]][location+'_attention_alignment'][:video_input_length[files[count]], :audio_input_length[files[count]]]
                        im = plt.imshow(img, vmin=0.0, vmax=1.0)
                        plt.title(title.upper())
                        if location == 'encoder':
                            plt.ylabel('video frame')
                            plt.xlabel('audio frame')
                        if location == 'decoder':
                            plt.ylabel('audio frame')
                            plt.xlabel('character')
                        cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
                        fig.colorbar(im, cax=cbar_ax)
                        os.makedirs('./visualization/'+location+'_attention_alignment_all/' + title + '/', exist_ok=True)
                        plt.savefig('./visualization/'+location+'_attention_alignment_all/' + title + '/' + str(count) + '.png',
                                    dpi=320)
                        plt.close(fig)
                        count += 1
                    eaa_max = np.max(eaa)
                    eaa_mean_max = np.mean([np.max(eaa[:, 0]) for i in range(len(eaa))])
                    eaa_max_list.append(eaa_max)
                    eaa_mean_max_list.append(eaa_mean_max)
            res[config_file.split('.')[0]] = {'max':np.mean(eaa_max_list), 'mean_max':np.mean(eaa_mean_max_list), 'cps':vcps+' '+acps+' '+dcps, 'snr':config['snr'], 'sl':skip_layer}
    print(','.join(['CPS','SL','SNR','Max','Mean Max']))
    for key,value in sorted(res.items()):
        print(','.join([str(value['cps']), str(value['sl']), str(value['snr']), str(value['max']), str(value['mean_max'])]))

def getRandomFiles(numberFiles):
    config = json.load(open('./configs/analyze/attention/mvlrs_v1_clean_av_align_n_0E+00_0E+00_0E+00_131_exp2.json', 'r'))
    experiment_path = config['experiment_path']
    experiment_name = config['experiment_name']
    x = p.load(open('./eval_data/' + experiment_path + experiment_name + '/eval_data_T_24_e1112.p', 'rb'))
    files = list(x.keys())
    files.remove('flops')
    random_files = [files[i] for i in np.random.randint(len(files), size=numberFiles)]
    return random_files

def store_attention_alignment_av_align():
    random_files = getRandomFiles(5)
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    label_length = p.load(open('./datasets/mvlrs_v1/label_length.p', 'rb'))
    rows, columns = 6, 6
    eaas = {}
    daas = {}
    for random_file in random_files:
        eaas[random_file] = []
        daas[random_file] = []
        for config_file in os.listdir('./configs/analyze/attention/'):
            config = json.load(open('./configs/finished/' + config_file, 'r'))
            experiment_path = config['experiment_path']
            experiment_name = config['experiment_name']
            print(experiment_name)
            if config['architecture'] == 'av_align' and config['dataset'] == 'mvlrs_v1' and (
                    'clean' in config['snr'] ):#or 'cafe_10db' in config['snr']
                dataset, noise, arch, skip_layer, vcps, acps, dcps, _, _, _ = experiment_path.split('/')
                x = p.load(open('./eval_data/' + experiment_path + experiment_name + '/eval_data_T_24_e1112.p', 'rb'))
                #print(x[random_file]['decoder_attention_alignment'].shape)
                #print(video_input_length[random_file])
                #print(audio_input_length[random_file])
                #print(label_length[random_file]['length'])
                eaa = x[random_file]['encoder_attention_alignment'][:video_input_length[random_file],:audio_input_length[random_file]]
                daa = x[random_file]['decoder_attention_alignment'][:audio_input_length[random_file],:label_length[random_file]['length']]
                eaas[random_file].append([eaa, experiment_path])
                daas[random_file].append([daa, experiment_path])
    for random_file, values in eaas.items():
        fig = plt.figure(figsize=(16, 16))
        for i, value in enumerate(values):
            v, exp = value
            fig.add_subplot(rows, columns, i+1)
            fig.subplots_adjust(hspace=0.9)
            plt.imshow(v, vmin=0.0, vmax=1.0, cmap='binary')
            dataset, noise, arch, skip_layer, vcps, acps, dcps, _, _, _ = exp.split('/')
            plt.title(skip_layer+' '+vcps+' '+acps+' '+dcps)
        #plt.show()
        plt.savefig('./visualization/attentionAlignment/encoder/' + random_file.replace('/','-') + '.png',
                    dpi=320)
        plt.close(fig)
    for random_file, values in daas.items():
        fig = plt.figure(figsize=(16, 16))
        for i, value in enumerate(values):
            v, exp = value
            fig.add_subplot(rows, columns, i+1)
            fig.subplots_adjust(hspace=0.9)
            plt.imshow(v, vmin=0.0, vmax=1.0, cmap='binary')
            dataset, noise, arch, skip_layer, vcps, acps, dcps, _, _, _ = exp.split('/')
            plt.title(skip_layer+' '+vcps+' '+acps+' '+dcps)
        #plt.show()
        plt.savefig('./visualization/attentionAlignment/decoder/' + random_file.replace('/','-') + '.png',
                    dpi=320)
        plt.close(fig)
        
def store_attention_alignment_bimodal():
    random_files = getRandomFiles(5)
    audio_input_length = p.load(open('./datasets/mvlrs_v1/audio_seq_len.p', 'rb'))
    video_input_length = p.load(open('./datasets/mvlrs_v1/video_seq_len.p', 'rb'))
    label_length = p.load(open('./datasets/mvlrs_v1/label_length.p', 'rb'))
    rows, columns = 7, 7
    eaas = {}
    daas = {}
    for random_file in random_files:
        eaas[random_file] = []
        daas[random_file] = []
        for config_file in os.listdir('./configs/analyze/bimodal_decoder/'):
            config = json.load(open('./configs/finished/' + config_file, 'r'))
            experiment_path = config['experiment_path']
            experiment_name = config['experiment_name']
            print(experiment_name)
            if config['architecture'] == 'bimodal' and config['dataset'] == 'mvlrs_v1' and (
                    'clean' in config['snr']):#or 'cafe_10db' in config['snr']
                dataset, noise, arch, skip_layer, vcps, acps, vdcps, adcps, _, _, _ = experiment_path.split('/')
                x = p.load(open('./eval_data/' + experiment_path + experiment_name + '/eval_data_T_24_e1112.p', 'rb'))
                #print(x[random_file]['decoder_attention_alignment'].shape)
                #print(video_input_length[random_file])
                #print(audio_input_length[random_file])
                #print(label_length[random_file]['length'])
                eaa = x[random_file]['decoder_attention_alignment_video'][:video_input_length[random_file],:label_length[random_file]['length']]
                daa = x[random_file]['decoder_attention_alignment_audio'][:audio_input_length[random_file],:label_length[random_file]['length']]
                eaas[random_file].append([eaa, experiment_path])
                daas[random_file].append([daa, experiment_path])
    for random_file, values in eaas.items():
        fig = plt.figure(figsize=(16, 16))
        for i, value in enumerate(values):
            v, exp = value
            fig.add_subplot(rows, columns, i+1)
            fig.subplots_adjust(hspace=0.9)
            plt.imshow(v, vmin=0.0, vmax=1.0, cmap='gray')
            dataset, noise, arch, skip_layer, vcps, acps, vdcps, adcps, _, _, _ = exp.split('/')
            plt.title(skip_layer+' '+vcps+' '+acps+'\n'+vdcps+' '+adcps)
        #plt.show()
        plt.savefig('./visualization/attentionAlignmentBimodal/decoderVideo/' + random_file.replace('/','-') + '.png',
                    dpi=320)
        plt.close(fig)
    for random_file, values in daas.items():
        fig = plt.figure(figsize=(16, 16))
        for i, value in enumerate(values):
            v, exp = value
            fig.add_subplot(rows, columns, i+1)
            fig.subplots_adjust(hspace=0.9)
            plt.imshow(v, vmin=0.0, vmax=1.0, cmap='gray')
            dataset, noise, arch, skip_layer, vcps, acps, vdcps, adcps, _, _, _ = exp.split('/')
            plt.title(skip_layer+' '+vcps+' '+acps+'\n'+vdcps+' '+adcps)
        #plt.show()
        plt.savefig('./visualization/attentionAlignmentBimodal/decoderAudio/' + random_file.replace('/','-') + '.png',
                    dpi=320)
        plt.close(fig)
        
store_attention_alignment_bimodal()