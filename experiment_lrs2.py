import os
import json
from pprint import pprint
from avsr import run_experiment

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main(config):
    experiment_name = config['experiment_name']
    dataset_name = config['dataset_name']
    tfrecords_path = 'N:/datasets/'+dataset_name+'/tfrecords/'
    
    video_train_record = tfrecords_path +'rgb36lips_train.tfrecord'
    video_trainTest_record = tfrecords_path +'rgb36lips_trainTest.tfrecord'
    video_test_record = tfrecords_path +'rgb36lips_test.tfrecord'
    labels_train_record = tfrecords_path +'characters_train.tfrecord'
    labels_trainTest_record = tfrecords_path +'characters_trainTest.tfrecord'
    labels_test_record = tfrecords_path +'characters_test.tfrecord'
    unit_list_file = 'F:/Documents/datasets/'+dataset_name+'/misc/character_list'

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
        architecture='av_align',
        logfile=dataset_name+'/'+experiment_name,
        unit_list_file=unit_list_file,
        loss_fun=config['loss_fun'],
        cell_type=config['cell_type'],
        use_dropout=config['use_dropout'],
        dropout_probability=config['dropout_probability'],
        encoder_units_per_layer=config['encoder_units_per_layer'],
        cost_per_sample=config['cost_per_sample'],
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        batch_size=config['batch_size'],
        write_attention_alignment=True,
        max_label_length=config['max_label_length'],
        separate_skip_rnn=config['separate_skip_rnn'],
        warmup_epochs=config['warmup_epochs'],
        warmup_max_len=config['warmup_max_len'],
        decoder_units_per_layer=config['decoder_units_per_layer'],
        lr_decay=config['lr_decay'],
        attention_type=config['attention_type'],
        write_summary=config['write_summary'],
    )

if __name__ == '__main__':
    exp_nr = 2
    for _ in range(1):
        for cost_per_sample in [0.0]: #0.0, 0.001, 0.0001, 0.00001
            for audio_encoder_layers in [(256, 256, 256)]:
                config = {'experiment_name': f'exp{exp_nr}_att_type_b',#f'exp{exp_nr}_a_skip_comb',f'exp{exp_nr}_normal_drop'
                          'dataset_name': 'mvlrs_v1',
                          'cost_per_sample': cost_per_sample,
                          'cell_type': ['skip_lstm', 'skip_lstm', 'lstm'],
                          'loss_fun':None,
                          'use_dropout':True,
                          'dropout_probability':(0.9, 0.9, 0.9),
                          'max_label_length':100,# LRS3 150, LRS2 100
                          'iterations':(100, 20),
                          'warmup_epochs':0,
                          'warmup_max_len':0,
                          'lr_decay':None,
                          'separate_skip_rnn':False,
                          'learning_rate':(0.001, 0.0001),
                          'batch_size':(40, 40),
                          'encoder_units_per_layer': ((256, ), audio_encoder_layers),
                          'decoder_units_per_layer':(256,),
                          'write_summary':False,
                          'attention_type':(('scaled_luong',)*1, ('scaled_luong',)*1)}
                config_path = f'./configs/{config["dataset_name"]}/{config["experiment_name"]}.json'
                if os.path.exists(config_path):
                    exp_nr += 1
                    print(exp_nr)
                else:
                    with open(config_path, 'w') as f:
                        json.dump(config, f)
                    pprint(config)
                    main(config)
                    exp_nr += 1