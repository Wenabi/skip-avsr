import os
import sys
import json
import time
import datetime
from avsr import run_experiment
from os import path

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main(config):
    dataset_name = config['dataset']
    tfrecords_path = 'N:/datasets/'+dataset_name+'/tfrecords/' #N:
    
    video_train_record = tfrecords_path +'rgb36lips_train.tfrecord'
    video_trainTest_record = tfrecords_path +'rgb36lips_trainTest.tfrecord'
    video_test_record = tfrecords_path +'rgb36lips_test.tfrecord'
    labels_train_record = tfrecords_path +'characters_train.tfrecord'
    labels_trainTest_record = tfrecords_path +'characters_trainTest.tfrecord'
    labels_test_record = tfrecords_path +'characters_test.tfrecord'
    unit_list_file = 'F:/Documents/datasets/'+dataset_name+'/misc/character_list' #F:/Documents

    audio_train_records = (
        tfrecords_path +'logmel_train_clean.tfrecord',
        #tfrecords_path +'logmel_train_cafe_10db.tfrecord',
        #tfrecords_path +'logmel_train_cafe_0db.tfrecord',
        #tfrecords_path +'logmel_train_cafe_-5db.tfrecord'
    )

    audio_trainTest_records = (
        tfrecords_path +'logmel_trainTest_clean.tfrecord',
        #tfrecords_path +'logmel_trainTest_cafe_10db.tfrecord',
        #tfrecords_path +'logmel_trainTest_cafe_0db.tfrecord',
        #tfrecords_path +'logmel_trainTest_cafe_-5db.tfrecord'
    )

    audio_test_records = (
       tfrecords_path +'logmel_test_clean.tfrecord',
       #tfrecords_path +'logmel_test_cafe_10db.tfrecord',
       #tfrecords_path +'logmel_test_cafe_0db.tfrecord',
       #tfrecords_path +'logmel_test_cafe_-5db.tfrecord'
    )

    iterations = (
        config['iterations'],  # clean
        #(250, 20),  # 10db
        #(250, 20),  # 0db
        #(250, 20)     # -5db
    )

    learning_rates = (
        config['learning_rate'],  # clean  (0.001, 0.0001)
        #(0.0005, 0.0001),  # 10db   (0.001, 0.0001)
        #(0.0005, 0.0001),  # 0db    (0.001, 0.0001)
        #(0.0005, 0.0001)   # -5db   (0.001, 0.0001)
    )


    run_experiment(
        video_train_record=video_train_record,
        video_trainTest_record=video_trainTest_record,
        video_test_record=video_test_record,
        labels_train_record=labels_train_record,
        labels_trainTest_record=labels_trainTest_record,
        labels_test_record=labels_test_record,
        audio_train_records=audio_train_records,
        audio_trainTest_records=audio_trainTest_records,
        audio_test_records=audio_test_records,
        iterations=iterations,
        learning_rates=learning_rates,
        architecture=config['architecture'],
        logfile=config['experiment_path']+config['experiment_name'],
        unit_list_file=unit_list_file,
        cell_type=config['cell_type'],
        encoder_units_per_layer=config['encoder_units_per_layer'],
        cost_per_sample=config['cost_per_sample'],
        experiment_name=config['experiment_name'],
        experiment_path=config['experiment_path'],
        dataset_name=dataset_name,
        batch_size=config['batch_size'],
        write_attention_alignment=True,
        max_label_length=config['max_label_length'],
        decoder_units_per_layer=config['decoder_units_per_layer'],
        write_summary=config['write_summary'],
        set_data_null=config['set_data_null'],
    )

if __name__ == '__main__':
    argv = sys.argv[1:]
    argv = {argv[i]:argv[i+1] for i in range(0,len(argv),2)}
    if len(argv) == 0:
        argv['-g'] = '0'
        argv['-d'] = 'mvlrs_v1'
    os.environ['CUDA_VISIBLE_DEVICES'] = argv['-g']
    dataset, gpu_num = None, None
    for config_file in os.listdir('./configs/'+argv['-d']+'/gpu_'+argv['-g']+'/'):
        print(config_file)
        config = json.load(open('./configs/'+argv['-d']+'/gpu_'+argv['-g']+'/'+config_file, 'r'))
        full_logfile = path.join('./logs', config['experiment_path'] + config['experiment_name'])
        with open(full_logfile, 'a') as f:
            f.write('Experiment Start:' + str(time.time()) + '\n')
        main(config)
        with open(full_logfile, 'a') as f:
            f.write('Experiment End:' + str(time.time()) + '\n')
        os.rename('./configs/'+argv['-d']+'/gpu_'+argv['-g']+'/'+config_file, './configs/'+argv['-d']+'/finished/'+config_file)
