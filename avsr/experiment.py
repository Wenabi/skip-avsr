from avsr import AVSR
from os import path, listdir
import tensorflow as tf


def run_experiment(
        video_train_record=None,
        video_trainTest_record=None,
        video_test_record=None,
        labels_train_record=None,
        labels_trainTest_record=None,
        labels_test_record=None,
        audio_train_records=None,
        audio_trainTest_records=None,
        audio_test_records=None,
        unit='character',
        unit_list_file='./avsr/misc/character_list',
        iterations=None,
        learning_rates=None,
        architecture='unimodal',
        logfile='tmp_experiment',
        warmup_epochs=0,
        warmup_max_len=0,
        mode='train',
        experiment_path=None,
        **kwargs):
    print(mode)
    if architecture == 'unimodal':
        video_processing = None
    else:
        video_processing = 'resnet_cnn'

    full_logfile = path.join('./logs', logfile)
    print('full_logfile', full_logfile)

    ## warmup on short sentences
    if warmup_epochs > 1 and mode == 'train':
        experiment = AVSR(
            unit=unit,
            unit_file=unit_list_file,
            audio_processing='features',
            audio_train_record=audio_train_records[0],
            audio_trainTest_record=audio_trainTest_records[0],
            audio_test_record=audio_test_records[0],
            video_processing=video_processing,
            video_train_record=video_train_record,
            video_trainTest_record=video_trainTest_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_trainTest_record=labels_trainTest_record,
            labels_test_record=labels_test_record,
            architecture=architecture,
            learning_rate=learning_rates[0][0],
            max_sentence_length=warmup_max_len,
            experiment_path=experiment_path,
            **kwargs
        )

        with open(full_logfile, 'a') as f:
            f.write('Warm up on short sentences for {} epochs \n'.format(warmup_max_len))

        experiment.train(
            logfile=full_logfile,
            num_epochs=warmup_epochs,
            try_restore_latest_checkpoint=True
        )

        with open(full_logfile, 'a') as f:
            f.write(5 * '=' + '\n')
    ##

    for lr, iters, audio_train, audio_trainTest, audio_test in zip(learning_rates, iterations, audio_train_records, audio_trainTest_records, audio_test_records):
        running = True
        while running:
            try:
                with open(full_logfile, 'a') as f:
                    ad_name = audio_train.split('_')
                    if len(ad_name) == 3:
                        ad_name = ad_name[2].split('.')[0]
                    elif len(ad_name) == 4:
                        ad_name = ad_name[2]+'_'+ad_name[3].split('.')[0]
                    #f.write(f'Start training with {ad_name} audio data.\n')
                skip_first_training = False
                ad_name_reached = False
                print(ad_name)
                for line in open(full_logfile, 'r').read().splitlines():
                    if ad_name in line:
                        ad_name_reached = True
                    if ad_name_reached:
                        if 'Stopped training early.' in line:
                            print('Skipping first training.')
                            skip_first_training = True
                            break
                if not skip_first_training and mode == 'train':
                    experiment = AVSR(
                        unit=unit,
                        unit_file=unit_list_file,
                        audio_processing='features',
                        audio_train_record=audio_train,
                        audio_trainTest_record=audio_trainTest,
                        audio_test_record=audio_test,
                        video_processing=video_processing,
                        video_train_record=video_train_record,
                        video_trainTest_record=video_trainTest_record,
                        video_test_record=video_test_record,
                        labels_train_record=labels_train_record,
                        labels_trainTest_record=labels_trainTest_record,
                        labels_test_record=labels_test_record,
                        architecture=architecture,
                        learning_rate=lr[0],
                        patience=5,
                        experiment_path=experiment_path,
                        **kwargs
                    )
                    experiment.train(
                        logfile=full_logfile,
                        num_epochs=iters[0]+1,
                        try_restore_latest_checkpoint=True,
                    )
            
                    with open(full_logfile, 'a') as f:
                        f.write(5*'=' + '\n')
                
                experiment = AVSR(
                    unit=unit,
                    unit_file=unit_list_file,
                    audio_processing='features',
                    audio_train_record=audio_train,
                    audio_trainTest_record=audio_trainTest,
                    audio_test_record=audio_test,
                    video_processing=video_processing,
                    video_train_record=video_train_record,
                    video_trainTest_record=video_trainTest_record,
                    video_test_record=video_test_record,
                    labels_train_record=labels_train_record,
                    labels_trainTest_record=labels_trainTest_record,
                    labels_test_record=labels_test_record,
                    architecture=architecture,
                    learning_rate=lr[1],
                    patience=10,
                    required_grahps=('train', 'eval') if mode == 'train' else ('eval'),
                    experiment_path=experiment_path,
                    **kwargs
                )
                if mode == 'train':
                    experiment.train(
                        logfile=full_logfile,
                        num_epochs=iters[1]+1,
                        try_restore_latest_checkpoint=True,
                    )
            
                    with open(full_logfile, 'a') as f:
                        f.write(20*'=' + '\n')
                    running = False
                elif mode == 'evaluate':
                    checkpoint_dir = path.join('checkpoints/' + experiment_path,
                                               path.split(logfile)[-1] + '/')
                    latest_ckp = tf.train.latest_checkpoint(checkpoint_dir)
                    modes = ['evaluateAllData', 'evaluateTrain']
                    experiment.evaluate(latest_ckp, modes, 1111) # 1111: number for test epoch to get eval_data
                    print('evaluated')
                    running = False
            except Exception as e:
                if mode == 'train':
                    print('Error restarting experiment.')
                    with open(full_logfile, 'a') as f:
                        f.write('Error restarting experiment.\n')
                else:
                    print(e)
                    exit()



def run_experiment_mixedsnrs(
        video_train_record=None,
        video_test_record=None,
        labels_train_record=None,
        labels_test_record=None,
        audio_train_record=None,
        audio_test_record=None,
        unit='character',
        unit_list_file='./avsr/misc/character_list',
        iterations=None,
        learning_rates=None,
        architecture='unimodal',
        logfile='tmp_experiment',
        **kwargs):

    if architecture == 'unimodal':
        video_processing = None
    else:
        video_processing = 'resnet_cnn'

    full_logfile = path.join('./logs', logfile)


    for lr, iters in zip(learning_rates, iterations):
        experiment = AVSR(
            unit=unit,
            unit_file=unit_list_file,
            audio_processing='features',
            audio_train_record=audio_train_record,
            audio_test_record=audio_test_record,
            video_processing=video_processing,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
            architecture=architecture,
            learning_rate=lr[0],
            **kwargs
        )
        experiment.train(
            logfile=full_logfile,
            num_epochs=iters[0]+1,
            try_restore_latest_checkpoint=True
        )

        with open(full_logfile, 'a') as f:
            f.write(5*'=' + '\n')

        experiment = AVSR(
            unit=unit,
            unit_file=unit_list_file,
            audio_processing='features',
            audio_train_record=audio_train_record,
            audio_test_record=audio_test_record,
            video_processing=video_processing,
            video_train_record=video_train_record,
            video_test_record=video_test_record,
            labels_train_record=labels_train_record,
            labels_test_record=labels_test_record,
            architecture=architecture,
            learning_rate=lr[1],
            **kwargs
        )
        experiment.train(
            logfile=full_logfile,
            num_epochs=iters[1]+1,
            try_restore_latest_checkpoint=True
        )

        with open(full_logfile, 'a') as f:
            f.write(20*'=' + '\n')

