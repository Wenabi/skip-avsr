import os
import json

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
