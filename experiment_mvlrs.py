import avsr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR

def main():

    experiment = avsr.AVSR(
        unit='character',
        unit_file='/run/media/john_tukey/download/datasets/MV-LRS/misc/character_list',
        video_processing='resnet_cnn',
        cnn_filters=(8, 16, 24, 32),
        cnn_dense_units=128,
        video_train_record='/run/media/john_tukey/download/datasets/MV-LRS/tfrecords/rgb_train_success.tfrecord',
        video_test_record='/run/media/john_tukey/download/datasets/MV-LRS/tfrecords/rgb_train_success.tfrecord',
        audio_processing='features',
        audio_train_record='/run/media/john_tukey/download/datasets/MV-LRS/tfrecords/logmel_train_success_10db.tfrecord',
        audio_test_record='/run/media/john_tukey/download/datasets/MV-LRS/tfrecords/logmel_train_success_10db.tfrecord',
        labels_train_record='/run/media/john_tukey/download/datasets/MV-LRS/tfrecords/characters_train_success.tfrecord',
        labels_test_record='/run/media/john_tukey/download/datasets/MV-LRS/tfrecords/characters_train_success.tfrecord',
        encoder_type='unidirectional',
        decoding_algorithm='beam_search',
        encoder_units_per_layer=(256, 256, 256),
        decoder_units_per_layer=(256, ),
        attention_type=(('scaled_luong', )*3, ('scaled_luong', )*1),
        optimiser='AMSGrad',
        batch_size=(64, 256),
        learning_rate=0.0001,
        label_skipping=False,
    )

    error = experiment.evaluate(
        checkpoint_path='./checkpoints/mvlrs_chars_r31_success_3x256_option2_AMSGrad/checkpoint.ckp-71',
        alignments_outdir='./alignments/tmp/',
    )
    for (k, v) in error.items():
        print(k + ': {:.4f}% '.format(v * 100))
    return

    # experiment.train(
    #     num_epochs=151,
    #     logfile='./logs/mvlrs_chars_r11_success_3x256_option2_AMSGrad',
    #     try_restore_latest_checkpoint=True
    # )



if __name__ == '__main__':
    main()
