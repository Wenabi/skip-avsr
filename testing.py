import os
import json

def changeExperimentNames():
    count = 1
    for i, zeros in enumerate(['5', '4', '3']):
        for j in range(5):
            exp_nr = (i+1)+j*3
            file_name = f'exp{exp_nr}_a_skip_cps'+zeros
            new_file_name = f'exp{count}_a_skip_cps'+zeros
            os.rename('./configs/mvlrs_v1/'+file_name+'.json','./configs/mvlrs_v1/'+new_file_name+'.json')
            config = json.load(open('./configs/mvlrs_v1/'+new_file_name+'.json', 'r'))
            config['experiment_name'] = new_file_name
            json.dump(config, open('./configs/mvlrs_v1/'+new_file_name+'.json', 'w'))
            os.rename('./eval_data/mvlrs_v1/' + file_name, './eval_data/mvlrs_v1/' + new_file_name)
            os.rename('./logs/mvlrs_v1/' + file_name, './logs/mvlrs_v1/' + new_file_name)
            os.rename('./checkpoints/mvlrs_v1/' + file_name, './checkpoints/mvlrs_v1/' + new_file_name)
            os.rename('./summaries/train/logs/mvlrs_v1/' + file_name, './summaries/train/logs/mvlrs_v1/' + new_file_name)
            print(exp_nr, zeros)
            count += 1
            

def evalData():
    import pickle as p
    file_path = f'./eval_data/mvlrs_v1/exp15_a_skip_cps3/eval_data_e102.p'
    file = p.load(open(file_path, 'rb'))
    avg_len = []
    print(len(file.keys()))
    avg_sum = []
    for key in file.keys():
        avg_len.append(len(file[key]['audio_updated_states']))
        avg_sum.append(sum(file[key]['audio_updated_states'])[0])
    import numpy as np
    print(np.mean(avg_len))
    print(np.mean(avg_sum))
    print(avg_len.count(115))
    
    
def viewTFRecord(dataset):
    import tensorflow as tf
    import pickle as p
    res = {}
    for tfrecord in ['N:/datasets/' + dataset + '/tfrecords/'+'logmel_train_clean.tfrecord','N:/datasets/' + dataset + '/tfrecords/'+'logmel_test_clean.tfrecord']:
        for example in tf.python_io.tf_record_iterator(tfrecord):
            filename = tf.train.Example.FromString(example).features.feature['filename'].bytes_list.value[0].decode('utf-8')
            input_length = tf.train.Example.FromString(example).features.feature['input_length'].int64_list.value[0]
            res[filename] = input_length
    p.dump(res, open('./datasets/'+dataset+'/audio_seq_len.p', 'wb'))
    
def viewTFRecord2(dataset):
    import tensorflow as tf
    import pickle as p
    res = {}
    for tfrecord in ['N:/datasets/' + 'Grid' + '/tfrecords/'+'rgb36lips_train.tfrecord','N:/datasets/' + 'Grid' + '/tfrecords/'+'rgb36lips_test.tfrecord']:
        print(tfrecord)
        for example in tf.python_io.tf_record_iterator(tfrecord):
            filename = tf.train.Example.FromString(example).features.feature['filename'].bytes_list.value[0].decode('utf-8')
            input_length = tf.train.Example.FromString(example).features.feature['input_length'].int64_list.value[0]
            res[filename] = input_length
    p.dump(res, open('./datasets/Grid/video_seq_len.p', 'wb'))

def masking():
    import numpy as np
    us = np.ones((48,55))
    print(us)
    length = np.random.randint(30,40,48)
    print(length)
    correct_sum = [np.sum(us[i][:length[i]]) for i in range(len(us))]
    print(correct_sum)

viewTFRecord()
viewTFRecord2()