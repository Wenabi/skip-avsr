from os import path
import os
from avsr.dataset_writer import TFRecordWriter
from avsr.utils import get_files

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main():
    dataset_name = 'mvlrs_v1'
    dataset_dir = f'F:/Documents/datasets/{dataset_name}/'
    train_list = f'F:/Documents/datasets/{dataset_name}/splits/train2.scp'
    trainTest_list = f'F:/Documents/datasets/{dataset_name}/splits/trainTest.scp'
    test_list = f'F:/Documents/datasets/{dataset_name}/splits/test2.scp'

    train = get_files(train_list, dataset_dir)
    trainTest = get_files(trainTest_list, dataset_dir)
    test = get_files(test_list, dataset_dir)

    label_map = dict()
    for file in train+trainTest+test:#+trainTest+test:
        label_map[path.splitext(file)[0]] = path.splitext(file.split(f'{dataset_name}/')[-1])[0]

    writer = TFRecordWriter(
        train_files=train,
        trainTest_files=trainTest,
        test_files=test,
        label_map=label_map,
        )
    #print('Writing labels records')
    #writer.write_labels_records(
    #    unit='character',
    #    unit_list_file=f'F:/Documents/datasets/{dataset_name}/misc/character_list',
    #    label_file=f'F:/Documents/datasets/{dataset_name}/configs/character_labels',
    #    train_record_name=f'N:/datasets/{dataset_name}/tfrecords/characters_train.tfrecord',
    #    #trainTest_record_name=f'N:/datasets/{dataset_name}/tfrecords/characters_trainTest.tfrecord',
    #    #test_record_name=f'N:/datasets/{dataset_name}/tfrecords/characters_test.tfrecord',
    #)
    print('Writing audio records')
    writer.write_audio_records(
        content_type='feature',
        extension='mp4',
        transform='logmel_stack_w8s3',
        snr_list=[10], # ['clean', 10, 0, -5]
        target_sr=16000,
        noise_type='cafe', #cafe, street, zeroing
        train_record_name=f'N:/datasets/{dataset_name}/tfrecords/logmel_train',
        trainTest_record_name=f'N:/datasets/{dataset_name}/tfrecords/logmel_trainTest',
        test_record_name=f'N:/datasets/{dataset_name}/tfrecords/logmel_test',
    )
    #print('Writing bmp records')
    #writer.write_bmp_records(
    #    train_record_name=f'N:/datasets/{dataset_name}/tfrecords/rgb36lips_train.tfrecord',
    #    #trainTest_record_name=f'N:/datasets/{dataset_name}/tfrecords/rgb36lips_trainTest.tfrecord',
    #    #test_record_name=f'N:/datasets/{dataset_name}/tfrecords/rgb36lips_test.tfrecord',
    #    bmp_dir=f'N:/datasets/{dataset_name}/aligned_openface/',
    #    output_resolution=(36, 36),
    #    crop_lips=True,
    #)


if __name__ == '__main__':
    main()

