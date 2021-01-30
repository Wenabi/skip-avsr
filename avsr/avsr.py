import tensorflow as tf
import collections
from .io_utils import make_iterator_from_one_record, create_unit_dict, make_iterator_from_two_records
from .video import cnn_layers
from .audio import process_audio
from .seq2seq import Seq2SeqModel
import time
import pandas
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from os import makedirs, path, system
from .utils import compute_wer, write_sequences_to_labelfile, compute_measures
from .visualise.beam_search import create_html, copy_headers
from .visualise.visvis import write_frames, write_json
from pprint import pprint
from tensorflow.python.framework import graph_util

from .io_utils import BatchedData


class Model(collections.namedtuple("Model", ("data", "model", "initializer", "batch_size"))):
    pass


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    return flops, params

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')


class AVSR(object):
    def __init__(self,
                 unit,
                 unit_file=None,
                 video_processing=None,
                 video_train_record=None,
                 video_trainTest_record=None,
                 video_test_record=None,
                 audio_processing=None,
                 audio_train_record=None,
                 audio_trainTest_record=None,
                 audio_test_record=None,
                 labels_train_record=None,
                 labels_trainTest_record=None,
                 labels_test_record=None,
                 batch_size=(64, 64),
                 cnn_filters=(8, 16, 32, 64),
                 cnn_dense_units=128,
                 regress_aus=False,
                 batch_normalisation=True,
                 instance_normalisation=False,
                 input_dense_layers=(0,),
                 architecture='unimodal',
                 encoder_type='unidirectional',
                 highway_encoder=False,
                 residual_encoder=False,
                 cell_type='lstm',
                 recurrent_l2_regularisation=0.0001,
                 weight_decay=0.0001,
                 encoder_units_per_layer=((256, ), (256, 256, 256)),
                 decoder_units_per_layer=(256,),
                 encoder_weight_sharing=False,
                 enable_attention=True,
                 attention_type=(('scaled_luong',)*1, ('scaled_luong',)*1),
                 use_dropout=True,
                 audio_encoder_dropout_probability=(0.9, 0.9, 0.9),
                 video_encoder_dropout_probability=(0.9, 0.9, 0.9),
                 decoder_dropout_probability=(0.9, 0.9, 0.9),
                 embedding_size=128,
                 sampling_probability_outputs=0.1,
                 label_smoothing=0.0,
                 decoding_algorithm='beam_search',
                 beam_width=10,
                 max_sentence_length=None,
                 optimiser='Adam',
                 learning_rate=0.001,
                 patience=5,
                 lr_decay=None,
                 loss_fun=None,
                 clip_gradients=True,
                 max_gradient_norm=1.0,
                 num_gpus=1,
                 write_attention_alignment=False,
                 write_beam_search_graphs=False,
                 write_estimated_modality_lags=False,
                 precision='float32',
                 profiling=False,
                 required_grahps=('train', 'eval'),
                 cost_per_sample=[0.0, 0.0, 0.0],
                 experiment_name=None,
                 experiment_path=None,
                 dataset_name=None,
                 max_label_length=500,
                 separate_skip_rnn=False,
                 write_summary=False,
                 write_eval_data=False,
                 set_data_null='',
                 modes=['evaluateAllData', 'evaluateTrain'],
                 **kwargs,
                 ):
        r"""

        Args:
            unit: A string that represents the decoded linguistic unit, one of ('phoneme', 'viseme', 'character')
            unit_file: Path to a file storing the unit vocabulary, having one unit per line
            video_processing: Visual CNN front-end,
                one of ('features', 'resnet_cnn', '2dconv_cnn', '3dconv_cnn') or None
            video_train_record: Path to a training .tfrecord file of visual examples
            video_test_record: Path to a testing .tfrecord file of visual examples
            audio_processing: Acoustic front-end, one of ('features', 'wav') or None.
            audio_train_record: Path to a training .tfrecord file of acoustic examples
            audio_test_record: Path to a testing .tfrecord file of acoustic examples
            labels_train_record: Path to a training .tfrecord file of ground-truth transcriptions
            labels_test_record: Path to a training .tfrecord file of ground-truth transcriptions
            batch_size: Number of training examples/full sentences to be processed at once
            cnn_filters: List holding the number of convolutional filters, one list element per CNN layer
            cnn_dense_units: Output size of the CNN front-end, defines the visual feature size
            batch_normalisation: Boolean
            input_dense_layers: List holding the number of units of the optional fully-connected layers added after the
                feature extraction stage, one list element per layer
            architecture: One of ('unimodal', 'bimodal', 'av_align')
            encoder_type: RNN encoder type, one of ('unidirectional', 'bidirectional')
            highway_encoder: Boolean, optionally adds highway connections in the RNNs
            residual_encoder: Boolean, optionally adds residual connections in the RNNs
            cell_type: RNN cell type, one of ('lstm', 'gru'). See cells.py for more variants.
            recurrent_l2_regularisation: Boolean, L2 weight loss
            weight_decay: Float, Weight decay used by the AdamW optimiser.
            encoder_units_per_layer: List holding the number of RNN units in each encoding layer
            decoder_units_per_layer: List holding the number of RNN units in each decoding layer
            enable_attention: Boolean
            attention_type: Two tuples of strings in:
                ('luong', 'scaled_luong', 'bahdanau', 'normed_bahadau', 'scaled_monotonic_luong')
                Expected format:
                - unimodal decoder: (cross-modal attention types, decoder-encoder attention types)
                - bimodal decoder:  (decoder - audio encoder attention types, decoder - video encoder attention types)
                The 'cross-modal attention types' is only mandatory for the `av_align` architecture
            use_dropout: Boolean
            dropout_probability: RNN dropout, list of 3 floats representing the keep probability of the
                (input, state, output) units respectively
            embedding_size: Linguistic unit embedding size, 0 means one-hot encodings, use a positive int to learn an
                embedding matrix of size [vocabulary_size, embedding_size]
            sampling_probability_outputs: Probability of the current decoding prediction id to be used as next input
                instead of the ground truth unit id
            decoding_algorithm: One of ('greedy', 'beam_search')
            beam_width: Controls the beam size when using the `beam_search` algorithm
            optimiser: One of ('Adam', 'AMSGrad')
            learning_rate: Learning algorithm constant
            loss_fun: Loss function. Use None for the default cross-entropy sequence loss, or one of
                ('focal_loss', 'mc_loss) for alternative loss functions
            clip_gradients: Boolean
            max_gradient_norm: Gradient clipping constant, used when gradient clipping is enabled
            num_gpus: deprecated
            write_attention_alignment: Generates alignment images of the attention weights for visualisation purposes
            write_beam_search_graphs: Generates html files of the beam search graphs
            precision: One of ('float32', 'float16')
            profiling: Generates a runtime profile
        """
        self._unit = unit
        self._unit_dict = create_unit_dict(unit_file=unit_file)

        self._video_processing = video_processing
        self._video_train_record = video_train_record
        self._video_trainTest_record = video_trainTest_record
        self._video_test_record = video_test_record
        self._video_input_length = None
        self._audio_processing = audio_processing
        self._audio_train_record = audio_train_record
        self._audio_trainTest_record = audio_trainTest_record
        self._audio_test_record = audio_test_record
        self._audio_input_length = None
        self._labels_train_record = labels_train_record
        self._labels_trainTest_record = labels_trainTest_record
        self._labels_test_record = labels_test_record

        self._write_attention_alignment = write_attention_alignment
        self._write_beam_search_graphs = write_beam_search_graphs
        self._write_estimated_modality_lags = write_estimated_modality_lags
        self._required_graphs = required_grahps
        
        self._hparams = tf.contrib.training.HParams(
            unit_dict=self._unit_dict,
            unit_file=unit_file,
            vocab_size=len(self._unit_dict),
            batch_size=batch_size,
            video_processing=video_processing,
            audio_processing=audio_processing,
            max_label_length=max_label_length,  # max lens from tcdtimit, LRS3 150, LRS2 100
            max_sentence_length=max_sentence_length,
            batch_normalisation=batch_normalisation,
            instance_normalisation=instance_normalisation,
            input_dense_layers=input_dense_layers,
            encoder_type=encoder_type,
            architecture=architecture,
            highway_encoder=highway_encoder,
            residual_encoder=residual_encoder,
            regress_aus=regress_aus,
            cell_type=cell_type,
            recurrent_l2_regularisation=None if optimiser == 'AdamW' else recurrent_l2_regularisation,
            weight_decay=weight_decay,
            encoder_units_per_layer=encoder_units_per_layer,
            decoder_units_per_layer=decoder_units_per_layer,
            encoder_weight_sharing=encoder_weight_sharing,
            bijective_state_copy=False,  # feature under testing
            enable_attention=enable_attention,
            attention_type=attention_type,
            use_dropout=use_dropout,
            audio_encoder_dropout_probability=audio_encoder_dropout_probability,
            video_encoder_dropout_probability=video_encoder_dropout_probability,
            decoder_dropout_probability=decoder_dropout_probability,
            embedding_size=embedding_size,
            sampling_probability_outputs=sampling_probability_outputs,
            label_smoothing=label_smoothing,
            decoding_algorithm=decoding_algorithm,
            beam_width=beam_width,
            use_ctc=False,
            optimiser=optimiser,
            loss_scaling=128 if precision == 'float16' else 1,
            learning_rate=learning_rate,
            patience=patience,
            lr_decay=lr_decay,
            loss_fun=loss_fun,
            clip_gradients=clip_gradients,
            max_gradient_norm=max_gradient_norm,
            num_gpus=num_gpus,
            write_attention_alignment=write_attention_alignment,
            dtype=tf.float16 if precision == 'float16' else tf.float32,
            profiling=profiling,
            cost_per_sample=cost_per_sample,
            experiment_name=experiment_name,
            experiment_path=experiment_path,
            dataset_name=dataset_name,
            separate_skip_rnn=separate_skip_rnn,
            write_summary=write_summary,
            write_eval_data=write_eval_data,
            set_data_null=set_data_null,
            modes=modes,
            kwargs=kwargs,
        )
        
        self._hparams_audio = tf.contrib.training.HParams(
            frame_length_msec=25,  # 25 > 20
            frame_step_msec=10,
            sample_rate=16000,
            mel_lower_edge_hz=125,
            mel_upper_edge_hz=7600,  # 11025 > 7600
            num_mel_bins=80,  # 30 > 60 > 80
            num_mfccs=80,  # 26 > 13
        )

        self._hparams_video = tf.contrib.training.HParams(
            cnn_filters=cnn_filters,
            cnn_dense_units=cnn_dense_units,
        )

        self._create_graphs()
        self._create_models()
        self._create_sessions()
        self._initialize_sessions()
        
    def __del__(self):
        if 'train' in self._required_graphs:
            self._train_session.close()
        if 'eval' in self._required_graphs:
            self._evaluate_session.close()
        # self._predict_session.close()

    def train(self,
              logfile,
              num_epochs=400,
              try_restore_latest_checkpoint=False,
              ):

        checkpoint_dir = path.join('checkpoints/'+self._hparams.experiment_path, path.split(logfile)[-1]+'/')
        checkpoint_path = path.join(checkpoint_dir, 'checkpoint.ckp')
        makedirs(path.dirname(checkpoint_dir), exist_ok=True)
        makedirs(path.dirname(logfile), exist_ok=True)
        makedirs(path.dirname(logfile + '_info'), exist_ok=True)
        if self._hparams.write_eval_data:
            makedirs(
                path.dirname(f'./eval_data/{self._hparams.experiment_path}/{self._hparams.experiment_name}/'), exist_ok=True)
        if self._hparams.write_summary:
            self._initialize_summaries('summaries', logfile)
        
        #self._video_input_length = p.load(open('./datasets/'+self._hparams.dataset_name+'/video_seq_len.p', 'rb'))
        #self._audio_input_length = p.load(open('./datasets/'+self._hparams.dataset_name+'/audio_seq_len.p', 'rb'))
        
        last_epoch = 0
        if try_restore_latest_checkpoint is True:
            try:
                latest_ckp = tf.train.latest_checkpoint(checkpoint_dir)
                last_epoch = int(latest_ckp.split('-')[-1])
                self._train_model.model.saver.restore(
                    sess=self._train_session,
                    save_path=latest_ckp, )
                print('Restoring checkpoint from epoch {}\n'.format(last_epoch))
            except Exception:
                print('Could not restore from checkpoint, training from scratch!\n')

        loss_diff_list = []
        with open(logfile, 'r') as f:
            log_batch_loss = []
            log_validation_loss = []
            for line in f.read().splitlines():
                if 'batch_loss' in line:
                    log_batch_loss.append(float(line.split(' ')[-1]))
                elif 'validation_loss' in line:
                    log_validation_loss.append(float(line.split(' ')[-1]))
            for i in range(len(log_batch_loss)):
                loss_diff = log_validation_loss[i] - log_batch_loss[i]
                if loss_diff > 0.05:
                    loss_diff_list.append(loss_diff)
                if loss_diff < 0.05 and len(loss_diff_list) >= 1:
                    loss_diff_list = []
        print(f'Loss Diff List is {loss_diff_list}')
        if len(loss_diff_list) < self._hparams.patience:
            f = open(logfile, 'a')
            num_epochs = last_epoch+num_epochs if num_epochs <100 else num_epochs
            num_epochs = min(num_epochs, 125)
            print(last_epoch+1, num_epochs, self._hparams.experiment_name)
            for current_epoch in range(last_epoch+1, num_epochs):
                epoch = current_epoch
                print(f'Epoch: {epoch}', self._hparams.experiment_name)
                self._train_session.run([stream.iterator_initializer for stream in self._train_model.data
                                         if stream is not None])
                sum_loss = 0
                sum_gradient_norm = 0
                batches = 0
                batch_loss_list = []
                loss_rate_list = []
                budget_loss_list = []

                self._video_input_length = self._train_model.data[0].inputs_length
                self._audio_input_length = self._train_model.data[1].inputs_length

                if 'skip' in self._hparams.cell_type[0]:
                    video_update_states_rate = []
                    video_update_states = []
                    video_state_numbers = []
                    
                    us = self._train_model.model.video_skip_infos.updated_states
                    v_us = tf.reduce_sum(us, 1)
                    v_s_number = tf.reshape(self._video_input_length, (-1,1))
                    v_us_rate = tf.divide(v_us, tf.cast(v_s_number, tf.float32))
                if 'skip' in self._hparams.cell_type[1]:
                    audio_update_states_rate = []
                    audio_update_states = []
                    audio_state_numbers = []

                    us = self._train_model.model.audio_skip_infos.updated_states
                    a_us = tf.reduce_sum(us, 1)
                    a_s_number = tf.reshape(self._audio_input_length, (-1,1))
                    a_us_rate = tf.divide(a_us, tf.cast(a_s_number, tf.float32))
                if 'skip' in self._hparams.cell_type[2]:
                    if self._hparams.architecture == 'av_align':
                        decoder_update_states_rate = []
                        decoder_update_states = []
                        decoder_state_numbers = []

                        us = self._train_model.model.decoder_skip_infos.updated_states
                        d_us = tf.reduce_sum(us, 1)
                        d_s_number = tf.reshape(self._audio_input_length, (-1, 1))
                        d_us_rate = tf.divide(d_us, tf.cast(d_s_number, tf.float32))
                    elif self._hparams.architecture == 'bimodal':
                        decoder_update_states_rate_video = []
                        decoder_update_states_video = []
                        decoder_state_numbers_video = []
                        decoder_update_states_rate_audio = []
                        decoder_update_states_audio = []
                        decoder_state_numbers_audio = []

                        us_v = self._train_model.model._decoder_skip_infos_video.updated_states
                        d_us_v = tf.reduce_sum(us_v, 1)
                        d_s_number_v = tf.reshape(self._video_input_length, (-1, 1))
                        d_us_rate_v = tf.divide(d_us_v, tf.cast(d_s_number_v, tf.float32))

                        us_a = self._train_model.model._decoder_skip_infos_audio.updated_states
                        d_us_a = tf.reduce_sum(us_a, 1)
                        d_s_number_a = tf.reshape(self._audio_input_length, (-1, 1))
                        d_us_rate_a = tf.divide(d_us_a, tf.cast(d_s_number_a, tf.float32))
                
                start = time.time()
    
                try:
                    while True:
                        #session_outputs = [self._train_model.model.train_op,
                        #                   self._train_model.model.batch_loss,
                        #                   self._train_model.model.global_norm,
                        #                   tf.contrib.summary.all_summary_ops(),
                        #                   self._merged_train_summaries,
                        #                   self._train_model.model.global_step,
                        #                   self._train_model.model._loss_rate,
                        #                   self._train_model.model._budget_loss,
                        #                   self._train_model.model.current_lr,
                        #                   self._train_model.data[0].inputs_filenames]
                        
                        #out = self._train_session.run(session_outputs, **self.sess_opts)
                        
                        session_outputs = {
                            'train_op':self._train_model.model.train_op,
                            'batch_loss':self._train_model.model.batch_loss,
                            'global_norm':self._train_model.model.global_norm,
                            'all_summary_ops':tf.contrib.summary.all_summary_ops(),
                            'global_step':self._train_model.model.global_step,
                            'loss_rate':self._train_model.model._loss_rate,
                            'budget_loss':self._train_model.model._budget_loss,
                            'inputs_filenames':self._train_model.data[0].inputs_filenames
                        }

                        if 'skip' in self._hparams.cell_type[0]:
                            #session_outputs['video_updated_states'] = self._train_model.model.video_skip_infos.updated_states
                            session_outputs['video_updated_states'] = v_us
                            session_outputs['video_state_number'] = v_s_number
                            session_outputs['video_updated_states_rate'] = v_us_rate
                        if 'skip' in self._hparams.cell_type[1]:
                            #session_outputs['audio_updated_states'] = self._train_model.model.audio_skip_infos.updated_states
                            session_outputs['audio_updated_states'] = a_us
                            session_outputs['audio_state_number'] = a_s_number
                            session_outputs['audio_updated_states_rate'] = a_us_rate
                        if 'skip' in self._hparams.cell_type[2]:
                            if self._hparams.architecture == 'av_align':
                                session_outputs['decoder_updated_states'] = d_us
                                session_outputs['decoder_state_number'] = d_s_number
                                session_outputs['decoder_updated_states_rate'] = d_us_rate
                            elif self._hparams.architecture == 'bimodal':
                                session_outputs['decoder_updated_states_video'] = d_us_v
                                session_outputs['decoder_state_number_video'] = d_s_number_v
                                session_outputs['decoder_updated_states_rate_video'] = d_us_rate_v
                                session_outputs['decoder_updated_states_audio'] = d_us_a
                                session_outputs['decoder_state_number_audio'] = d_s_number_a
                                session_outputs['decoder_updated_states_rate_audio'] = d_us_rate_a
                        if self._hparams.write_summary:
                            session_outputs['merged_train_summaries'] = self._merged_train_summaries
                        
                        out_list = self._train_session.run(list(session_outputs.values()))
                        outputs = dict(zip(session_outputs.keys(), out_list))
                        
                        batch_loss_list.append(outputs['batch_loss'])
                        loss_rate_list.append(outputs['loss_rate'])
                        budget_loss_list.append(outputs['budget_loss'])
                        if 'skip' in self._hparams.cell_type[0]:
                            video_update_states_rate.append(np.mean(outputs['video_updated_states_rate']))
                            video_state_numbers.append(np.mean(outputs['video_state_number']))
                            video_update_states.append(np.mean(outputs['video_updated_states']))
                        if 'skip' in self._hparams.cell_type[1]:
                            audio_update_states_rate.append(np.mean(outputs['audio_updated_states_rate']))
                            audio_state_numbers.append(np.mean(outputs['audio_state_number']))
                            audio_update_states.append(np.mean(outputs['audio_updated_states']))
                        if 'skip' in self._hparams.cell_type[2]:
                            if self._hparams.architecture == 'av_align':
                                decoder_update_states_rate.append(np.mean(outputs['decoder_updated_states_rate']))
                                decoder_state_numbers.append(np.mean(outputs['decoder_state_number']))
                                decoder_update_states.append(np.mean(outputs['decoder_updated_states']))
                            elif self._hparams.architecture == 'bimodal':
                                decoder_update_states_rate_video.append(np.mean(outputs['decoder_updated_states_rate_video']))
                                decoder_state_numbers_video.append(np.mean(outputs['decoder_state_number_video']))
                                decoder_update_states_video.append(np.mean(outputs['decoder_updated_states_video']))
                                decoder_update_states_rate_audio.append(np.mean(outputs['decoder_updated_states_rate_audio']))
                                decoder_state_numbers_audio.append(np.mean(outputs['decoder_state_number_audio']))
                                decoder_update_states_audio.append(np.mean(outputs['decoder_updated_states_audio']))
                                
                        if self._hparams.profiling is True:
                            self.profiler.add_step(batches, self.run_meta)
    
                            from tensorflow.python.profiler import option_builder
    
                            self.profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
                                                                      .trainable_variables_parameter()))
    
                            opts = option_builder.ProfileOptionBuilder.time_and_memory()
                            self.profiler.profile_operations(options=opts)
    
                            opts = (option_builder.ProfileOptionBuilder(
                                option_builder.ProfileOptionBuilder.time_and_memory())
                                    .with_step(batches)
                                    .with_timeline_output('/tmp/timelines/').build())
    
                            self.profiler.profile_graph(options=opts)
                        
                        if self._hparams.write_summary:
                            self._train_summary_writer.add_summary(outputs['merged_train_summaries'], outputs['global_step'])
                        
                        batch_loss = outputs['batch_loss'][0] if type(outputs['batch_loss']) == np.ndarray else outputs['batch_loss']
                        sum_loss += batch_loss
                        sum_gradient_norm += outputs['global_norm'][0] if type(outputs['global_norm']) == np.ndarray else outputs['global_norm']
                        global_norm = outputs['global_norm']
                        print('batch: {}, batch loss: {:.2f}, gradient norm: {:.2f}'.format(batches, batch_loss, global_norm))
                        batches += 1
    
                except tf.errors.OutOfRangeError:
                    pass
    
                print('epoch time: {}'.format(time.time() - start))
                o = 'Average batch_loss as epoch {} is {}'.format(epoch, sum_loss / batches)
                print(o)
                f.write(o+'\n')
                o = 'Average gradient_norm as epoch {} is {}'.format(epoch, sum_gradient_norm / batches)
                print(o)
                f.write(o + '\n')
    
                output_info = 'Batch_Loss:{} Budget_Loss:{} Loss_Rate:{}'.format(
                    np.mean(batch_loss_list), np.mean(budget_loss_list), np.mean(loss_rate_list))
                print(output_info)
                f.write(output_info + '\n')
                f.flush()
                if 'skip' in self._hparams.cell_type[0]:
                    output_info = 'Video_Updated_States_Rate:{} Video_Updated_States:{} Video_Number_Of_States:{}'.format(
                        np.mean(video_update_states_rate), np.mean(video_update_states), np.mean(video_state_numbers))
                    print(output_info)
                    f.write(output_info + '\n')
                    f.flush()
                if 'skip' in self._hparams.cell_type[1]:
                    output_info = 'Audio_Updated_States_Rate:{} Audio_Updated_States:{} Audio_Number_Of_States:{}'.format(
                        np.mean(audio_update_states_rate), np.mean(audio_update_states), np.mean(audio_state_numbers))
                    print(output_info)
                    f.write(output_info+'\n')
                    f.flush()
                if 'skip' in self._hparams.cell_type[2]:
                    if self._hparams.architecture == 'av_align':
                        output_info = 'Decoder_Updated_States_Rate:{} Decoder_Updated_States:{} Decoder_Number_Of_States:{}'.format(
                            np.mean(decoder_update_states_rate), np.mean(decoder_update_states), np.mean(decoder_state_numbers))
                        print(output_info)
                        f.write(output_info+'\n')
                        f.flush()
                    elif self._hparams.architecture == 'bimodal':
                        output_info = 'Decoder_Updated_States_Rate_Video:{} Decoder_Updated_States_Video:{} Decoder_Number_Of_States_Video:{}'.format(
                            np.mean(decoder_update_states_rate_video), np.mean(decoder_update_states_video),
                            np.mean(decoder_state_numbers_video))
                        print(output_info)
                        f.write(output_info + '\n')
                        f.flush()
                        output_info = 'Decoder_Updated_States_Rate_Audio:{} Decoder_Updated_States_Audio:{} Decoder_Number_Of_States_Audio:{}'.format(
                            np.mean(decoder_update_states_rate_audio), np.mean(decoder_update_states_audio),
                            np.mean(decoder_state_numbers_audio))
                        print(output_info)
                        f.write(output_info + '\n')
                        f.flush()
    
                save_path = self._train_model.model.saver.save(
                    sess=self._train_session,
                    save_path=checkpoint_path,
                    global_step=epoch,
                )
                validation_loss = self.calc_validation_loss(save_path)
                print(f'Validation_Loss is: {validation_loss}')
                o = 'Average validation_loss as epoch {} is {}'.format(epoch, validation_loss)
                print(o)
                f.write(o + '\n')
                f.flush()
                
                if epoch % 1 == 0:
                    #if epoch % 5 == 0:
                    #    modes = ['evaluateAllData', 'evaluateTrain', 'evaluateAudio', 'evaluateVideo', 'evaluateNoData']
                    #else:
                    error_rates = self.evaluate(save_path, epoch)
                    error_rate = error_rates['evaluateAllData']
                    error_rate_train = error_rates['evaluateTrain']
                    #if epoch % 5 == 0:
                    #    error_rate_Audio = error_rates['evaluateAudio']
                    #    error_rate_Video = error_rates['evaluateVideo']
                    #    error_rate_NoData = error_rates['evaluateNoData']
                    #    evaluate_names = ['error_rate', 'error_rate_train', 'error_rate_Audio', 'error_rate_Video', 'error_rate_NoData']
                    #    data = np.array([list(e.values()) for e in
                    #                     [error_rate, error_rate_train, error_rate_Audio,
                    #                      error_rate_Video, error_rate_NoData]])
                    #else:
                    evaluate_names = ['error_rate', 'error_rate_train']
                    data = np.array([list(e.values()) for e in
                                     [error_rate, error_rate_train]])
                    print(pandas.DataFrame(data, evaluate_names, error_rate.keys()))
    
                    for (k, v) in error_rate.items():
                        f.write(k + ': {:.4f}% '.format(v * 100))
                    f.write('\n')
                    for (k, v) in error_rate_train.items():
                        f.write("train_" + k + ': {:.4f}% '.format(v * 100))
                    f.write('\n')
                    #if epoch % 5 == 0:
                    #    for (k, v) in error_rate_Audio.items():
                    #        f.write("A_" + k + ': {:.4f}% '.format(v * 100))
                    #    f.write('\n')
                    #    for (k, v) in error_rate_Video.items():
                    #        f.write("V_" + k + ': {:.4f}% '.format(v * 100))
                    #    f.write('\n')
                    #    for (k, v) in error_rate_NoData.items():
                    #        f.write("nD_" + k + ': {:.4f}% '.format(v * 100))
                    #    f.write('\n')
                    f.flush()
                #Early Stopping
                loss_diff = validation_loss - (sum_loss / batches)
                print(f'Loss Difference is {loss_diff}.')
                if loss_diff > 0.05:
                    loss_diff_list.append(loss_diff)
                if loss_diff < 0.05 and len(loss_diff_list) >= 1:
                    loss_diff_list = []
                if len(loss_diff_list) == self._hparams.patience:
                    print(f'Stopped training early. Using model of epoch {current_epoch}.')
                    f.write(f'Stopped training early. Using model of epoch {current_epoch}.\n')
                    f.flush()
                    if num_epochs < 100:
                        modes = ['evaluateAllData', 'evaluateTrain']
                        error_rates = self.evaluate(save_path, epoch)
                        error_rate = error_rates['evaluateAllData']
                        error_rate_train = error_rates['evaluateTrain']
                        evaluate_names = ['error_rate', 'error_rate_train']
                        data = np.array([list(e.values()) for e in
                                         [error_rate, error_rate_train]])
                        print(pandas.DataFrame(data, evaluate_names, error_rate.keys()))
    
                        for (k, v) in error_rate.items():
                            f.write(k + ': {:.4f}% '.format(v * 100))
                        f.write('\n')
                        for (k, v) in error_rate_train.items():
                            f.write("train_" + k + ': {:.4f}% '.format(v * 100))
                        f.write('\n')
                        f.flush()
                    break
                print(f'Loss Diff List is {loss_diff_list}')
            f.close()

    def evaluate(self,
                 checkpoint_path,
                 epoch=None,
                 alignments_outdir='./alignments/tmp/',
                 beam_graphs_outdir='./beam_graphs/tmp/',
                 ):
        print('evaluate_checkpoint_path', checkpoint_path)
        if self._hparams.write_eval_data:
            makedirs(
                path.dirname(f'./eval_data/{self._hparams.experiment_path}/{self._hparams.experiment_name}/'), exist_ok=True)
        error_rates = {}
        
        for mode in self._hparams.modes:#'evaluateAudio', 'evaluateVideo', 'evaluateNoData'
            print(mode)
            error_rates[mode] = None
            if mode in ['evaluateAllData', 'evaluateAllTrainData', 'evaluateTrain']:
                evaluate_data = {}
            sess = None
            evaluate_model = None
            if mode == 'evaluateAllData':
                sess = self._evaluate_session
                evaluate_model = self._evaluate_model
                graph = self._evaluate_graph
            #elif mode == 'evaluateAudio':
            #    sess = self._evaluateAudio_session
            #    evaluate_model = self._evaluateAudio_model
            #elif mode == 'evaluateVideo':
            #    sess = self._evaluateVideo_session
            #    evaluate_model = self._evaluateVideo_model
            #elif mode == 'evaluateNoData':
            #    sess = self._evaluateNoData_session
            #    evaluate_model = self._evaluateNoData_model
            elif mode == 'evaluateTrain':
                sess = self._evaluateTrain_session
                evaluate_model = self._evaluateTrain_model
                graph = self._evaluateTrain_graph
            elif mode == 'evaluateAllTrainData':
                sess = self._evaluateAllTrainData_session
                evaluate_model = self._evaluateAllTrainData_model
                graph = self._evaluateAllTrainData_graph
            evaluate_model.model.saver.restore(
                sess=sess,
                save_path=checkpoint_path
            )
    
            sess.run([stream.iterator_initializer for stream in evaluate_model.data
                      if stream is not None])
            predictions_dict = {}
            labels_dict = {}
            
            if self._hparams.video_processing is not None:
                data = evaluate_model.data[0]
            elif self._hparams.audio_processing is not None:
                data = evaluate_model.data[1]
            else:
                raise ValueError('At least one of A/V streams must be enabled')

            session_dict = {
                'predicted_ids': evaluate_model.model._decoder.inference_predicted_ids,
                'labels': data.labels,
                'input_filenames': data.inputs_filenames,
                'labels_filenames': data.labels_filenames,
                # 'encoder_outputs': self._evaluate_model.model._audio_encoder._encoder_outputs,
                # 'weights': self._evaluate_model.model._audio_encoder._encoder_cells.weights,
            }
            if self._hparams.write_eval_data is True and mode in ['evaluateAllData', 'evaluateAllTrainData',
                                                                  'evaluateTrain']:
                flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
                flops = flops.total_float_ops
                print(flops)
                evaluate_data['flops'] = flops
            
            if mode in ['evaluateAllData', 'evaluateAllTrainData', 'evaluateTrain']:
                if 'skip' in self._hparams.cell_type[0]:
                    session_dict['video_updated_states'] = evaluate_model.model.video_skip_infos.updated_states
                    session_dict['video_inputs_lengths'] = evaluate_model.data[0].inputs_length
                if 'skip' in self._hparams.cell_type[1]:
                    session_dict['audio_updated_states'] = evaluate_model.model.audio_skip_infos.updated_states
                    session_dict['audio_inputs_lengths'] = evaluate_model.data[1].inputs_length
                if 'skip' in self._hparams.cell_type[2]:
                    if self._hparams.architecture == 'av_align':
                        session_dict['decoder_updated_states'] = evaluate_model.model.decoder_skip_infos.updated_states
                        session_dict['audio_inputs_lengths'] = evaluate_model.data[1].inputs_length
                    elif self._hparams.architecture == 'bimodal':
                        session_dict['decoder_updated_states_video'] = evaluate_model.model._decoder_skip_infos_video.updated_states
                        session_dict['decoder_updated_states_audio'] = evaluate_model.model._decoder_skip_infos_audio.updated_states
                        session_dict['video_inputs_lengths'] = evaluate_model.data[0].inputs_length
                        session_dict['audio_inputs_lengths'] = evaluate_model.data[1].inputs_length
            
                if self._write_attention_alignment is True:
                    session_dict['decoder_attention_summary'] = evaluate_model.model._decoder.attention_summary
                    #for the bimodal architecture, attention_summary will return a list of two image summaries (A and V)
                    session_dict['decoder_attention_alignment'] = evaluate_model.model._decoder.attention_alignment
                    if self._write_estimated_modality_lags is True:
                        session_dict['video_input'] = evaluate_model.data[0].inputs
        
                    if self._hparams.architecture == 'av_align':
                        session_dict['encoder_attention_summary'] = evaluate_model.model._audio_encoder.attention_summary
                        if self._write_estimated_modality_lags:
                            pass
                        session_dict['encoder_attention_alignment'] = evaluate_model.model._audio_encoder.attention_alignment
        
                if self._write_beam_search_graphs is True:
                    session_dict['beam_search_output'] = evaluate_model.model._decoder.beam_search_output
                    copy_headers(out_dir=beam_graphs_outdir)

            eval_counter = 0
            while True:
                if eval_counter % 10 == 0:
                    print(mode, eval_counter)
                if eval_counter % 100 == 0 and eval_counter > 0:
                    if mode in ['evaluateAllTrainData'] and self._hparams.write_eval_data:
                        p.dump(evaluate_data, open(
                            f'./eval_data/{self._hparams.experiment_path}/{self._hparams.experiment_name}/eval_data_{mode[8:-4]}_{eval_counter}_e{epoch}.p',
                            'wb'))
                        evaluate_data = {}
                try:
                    #
                    out_list = sess.run(list(session_dict.values()))
                    outputs = dict(zip(session_dict.keys(), out_list))

                    #flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
                    #print("Flops: ", flops.total_float_ops)

                    #print('stats before freezing')
                    #flops1, params1 = stats_graph(graph)
                    #print([node.name for node in graph.as_graph_def().node if 'Decoder' in node.name and 'strided_slice_5' in node.name])
                    #output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [node.name for node in graph.as_graph_def().node if 'Decoder' in node.name and 'strided_slice_5' in node.name])
                    #with tf.gfile.GFile('graph.pb', "wb") as f:
                    #    f.write(output_graph.SerializeToString())

                    #with tf.Graph().as_default() as graph:
                    #    graph = load_pb('./graph.pb')
                    #    print('stats after freezing')
                    #    flops2 = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())

                    #print('GFLOPs: {}'.format(flops1.total_float_ops / 1000000000.0))
                    #print('GFLOPs: {}'.format(flops2.total_float_ops / 1000000000.0))

                    #exit()
                    
                    # debug time
                    # assert (any(list(out[2] == out[3])))
                    # assert (any(list(out[1] == out[3])))
                    
                    if mode == 'evaluateAllData':
                        if self._write_attention_alignment is True:
    
                            if self._hparams.architecture == 'unimodal':
                                dec_enc_summ = tf.Summary()
                                dec_enc_summ.ParseFromString(outputs['decoder_attention_summary'])
        
                            elif self._hparams.architecture == 'bimodal':
                                video_summary = tf.Summary()
                                video_summary.ParseFromString(outputs['decoder_attention_summary'][0])
        
                                audio_summary = tf.Summary()
                                audio_summary.ParseFromString(outputs['decoder_attention_summary'][1])
        
                            elif self._hparams.architecture == 'av_align':
                                dec_enc_summ = tf.Summary()
                                dec_enc_summ.ParseFromString(outputs['decoder_attention_summary'])
        
                                cross_modal_summary = tf.Summary()
                                cross_modal_summary.ParseFromString(outputs['encoder_attention_summary'])
                            else:
                                raise ValueError('Unknown architecture')
                    
                    for idx in range(len(outputs['input_filenames'])):  # could use batch_size here, but take care with the last smaller batch
                        predicted_ids = outputs['predicted_ids'][idx]
                        predicted_symbs = [self._unit_dict[sym] for sym in predicted_ids]
    
                        labels_ids = outputs['labels'][idx]
                        labels_symbs = [self._unit_dict[sym] for sym in labels_ids]
                        file = outputs['input_filenames'][idx].decode('utf-8')
                        
                        if self._hparams.write_eval_data is True and mode in ['evaluateAllData', 'evaluateAllTrainData', 'evaluateTrain']:
                            if self._hparams.architecture == 'av_align':
                                evaluate_data[file] = {
                                    'encoder_attention_summary': outputs['encoder_attention_summary'][idx],
                                    'encoder_attention_alignment': outputs['encoder_attention_alignment'][idx],
                                    'decoder_attention_summary': outputs['decoder_attention_summary'][idx],
                                    'decoder_attention_alignment': outputs['decoder_attention_alignment'][idx]}
                                if 'skip' in self._hparams.cell_type[0]:
                                    evaluate_data[file]['video_updated_states'] = outputs['video_updated_states'][idx][
                                                                                  :outputs['video_inputs_lengths'][idx]]
                                if 'skip' in self._hparams.cell_type[1]:
                                    evaluate_data[file]['audio_updated_states'] = outputs['audio_updated_states'][idx][
                                                                                  :outputs['audio_inputs_lengths'][idx]]
                                if 'skip' in self._hparams.cell_type[2]:
                                    evaluate_data[file]['decoder_updated_states'] = outputs['decoder_updated_states'][
                                        idx][:outputs['audio_inputs_lengths'][idx]]
                            elif self._hparams.architecture == 'bimodal':
                                evaluate_data[file] = {
                                    'decoder_attention_summary_video': outputs['decoder_attention_summary'][0][idx],
                                    'decoder_attention_summary_audio': outputs['decoder_attention_summary'][1][idx],
                                    'decoder_attention_alignment_video': outputs['decoder_attention_alignment'][0][idx],
                                    'decoder_attention_alignment_audio': outputs['decoder_attention_alignment'][1][idx]}
                                if 'skip' in self._hparams.cell_type[2]:
                                    evaluate_data[file]['decoder_updated_states_video'] = outputs['decoder_updated_states_video'][idx][:outputs['video_inputs_lengths'][idx]]
                                    evaluate_data[file]['decoder_updated_states_audio'] = outputs['decoder_updated_states_audio'][idx][:outputs['audio_inputs_lengths'][idx]]
                        
                        if self._write_attention_alignment is True and mode == 'evaluateAllData':
    
                            if self._hparams.architecture == 'unimodal':
                                fname = path.join(alignments_outdir, file + '.png')
                                makedirs(path.dirname(fname), exist_ok=True)
                                with tf.gfile.GFile(fname, mode='w') as img_f:
                                    img_f.write(dec_enc_summ.value[idx].image.encoded_image_string)
    
                            elif self._hparams.architecture == 'bimodal':
                                fname = path.join(alignments_outdir, file + '_video.png')
                                makedirs(path.dirname(fname), exist_ok=True)
                                with tf.gfile.GFile(fname, mode='w') as img_f:
                                    img_f.write(video_summary.value[idx].image.encoded_image_string)
    
                                fname = path.join(alignments_outdir, file + '_audio.png')
                                with tf.gfile.GFile(fname, mode='w') as img_f:
                                    img_f.write(audio_summary.value[idx].image.encoded_image_string)
    
                            elif self._hparams.architecture == 'av_align':
                                fname = path.join(alignments_outdir, file + '.png')
                                makedirs(path.dirname(fname), exist_ok=True)
                                with tf.gfile.GFile(fname, mode='w') as img_f:
                                    img_f.write(dec_enc_summ.value[idx].image.encoded_image_string)
    
                                fname = path.join(alignments_outdir, file + '_av.png')
                                with tf.gfile.GFile(fname, mode='w') as img_f:
                                    img_f.write(cross_modal_summary.value[idx].image.encoded_image_string)
                                
                                if self._write_estimated_modality_lags is True:
                                    import json
                                    av_sentence = outputs['encoder_attention_alignment'][idx][:, :, 0]
                                    at_sentence = outputs['decoder_attention_alignment'][idx][:, :, 0]
    
                                    from .visualise.modality_lags import write_fig, write_txt, write_praat_intensity,\
                                        get_at_timestamps, get_av_timestamps
                                    audio_stamps, tau = get_av_timestamps(av_sentence)
                                    fname = path.join(alignments_outdir, file + '_lags.png')
                                    write_fig(audio_stamps=audio_stamps, tau=tau, title=file, fname=fname)
                                    write_txt(audio_stamps=audio_stamps, tau=tau, fname=fname.replace('.png', '.txt'))
                                    write_praat_intensity(audio_stamps, tau, fname=fname.replace('.png', '.praat'))
    
                                    frames = outputs['video_input'][idx]
                                    png_dir = path.join(alignments_outdir, file + '_pngs')
                                    png_paths = write_frames(png_dir, frames)
                                    fname = path.join(alignments_outdir, file + '.meta.json')
    
                                    dataset_dir = '/run/media/john_tukey/download/datasets/lrs2/mvlrs_v1/main/'
                                    wav_name = path.join(alignments_outdir, file + '.wav')
                                    mp4_name = path.join(dataset_dir, file + '.mp4')
                                    spectrogram_name = path.join(alignments_outdir, file + '_spec.png')
                                    system('ffmpeg -i {} {}'.format(mp4_name, wav_name))
                                    system('sox {} -n rate 12k spectrogram -z 80 -r -m -l -y 65 -w Hann -o {}'.format(wav_name, spectrogram_name))
    
                                    write_json(
                                        av_sentence=av_sentence,
                                        at_sentence=at_sentence,
                                        labels=predicted_symbs,
                                        png_paths=png_paths,
                                        spectrogram=spectrogram_name,
                                        json_file=fname)
                            else:
                                raise ValueError('Unknown architecture')
    
                        if self._write_beam_search_graphs is True:
                            from .visualise.beam_search import create_html
    
                            predicted_ids_beams = outputs['beam_search_output'].predicted_ids[idx]
                            parent_ids_beams = outputs['beam_search_output'].parent_ids[idx]
                            scores_beams = outputs['beam_search_output'].scores[idx]
    
                            create_html(
                                predicted_ids=predicted_ids_beams,
                                parent_ids=parent_ids_beams,
                                scores=scores_beams,
                                labels_ids=labels_ids,
                                vocab=self._unit_dict,
                                filename=file,
                                output_dir=beam_graphs_outdir)
    
                        predictions_dict[file] = predicted_symbs
                        labels_dict[file] = labels_symbs
    
                except tf.errors.OutOfRangeError:
                    break
                eval_counter += 1

            recall, precision, fmeasure = compute_measures(predictions_dict, labels_dict)
            uer, uer_dict = compute_wer(predictions_dict, labels_dict)
            error_rate = {'c_error_rate': uer, "c_recall":recall, "c_precision":precision, "c_fmeasure":fmeasure}
            if self._unit == 'character':
                w_recall, w_precision, w_fmeasure = compute_measures(predictions_dict, labels_dict, split_words=True)
                wer, wer_dict = compute_wer(predictions_dict, labels_dict, split_words=True)
                error_rate['w_error_rate'] = wer
                error_rate['w_recall'] = w_recall
                error_rate['w_precision'] = w_precision
                error_rate['w_fmeasure'] = w_fmeasure

            outdir = path.join('predictions/' + self._hparams.experiment_path,
                               path.split(path.split(checkpoint_path)[0])[-1] + "/" + mode + "/")
            makedirs(outdir, exist_ok=True)
            info = mode.replace("evaluate", "")
            write_sequences_to_labelfile(predictions_dict,
                                         path.join(outdir, 'predicted_' + info + '_epoch_{}.mlf'.format(epoch)),
                                         labels_dict,
                                         uer_dict)
            # from .analyse.analyse import plot_err_vs_seq_len, compute_uer_confusion_matrix
            # plot_err_vs_seq_len(labels_dict, uer_dict, 'tmp.pdf')
            # mat = compute_uer_confusion_matrix(predictions_dict=predictions_dict, labels_dict=labels_dict, unit_dict=self._unit_dict)
            error_rates[mode] = error_rate
            if mode in ['evaluateAllData', 'evaluateAllTrainData', 'evaluateTrain'] and self._hparams.write_eval_data:
                p.dump(evaluate_data, open(f'./eval_data/{self._hparams.experiment_path}/{self._hparams.experiment_name}/eval_data_{mode[8:-4]}_{eval_counter}_e{epoch}.p', 'wb'))
        return error_rates

    def calc_validation_loss(self, checkpoint_path):

        self._valLoss_model.model.saver.restore(
            sess=self._valLoss_session,
            save_path=checkpoint_path
        )
        self._valLoss_session.run([stream.iterator_initializer for stream in self._valLoss_model.data
                  if stream is not None])
        validation_losses = []

        if self._hparams.video_processing is not None:
            data = self._valLoss_model.data[0]
        elif self._hparams.audio_processing is not None:
            data = self._valLoss_model.data[1]
        else:
            raise ValueError('At least one of A/V streams must be enabled')

        session_dict = {
            'input_filenames': data.inputs_filenames,
            'validation_loss': self._valLoss_model.model.batch_loss
        }

        eval_counter = 0
        while True:
            if eval_counter % 10 == 0:
                print(eval_counter)
            try:
                out_list = self._valLoss_session.run(list(session_dict.values()))
                outputs = dict(zip(session_dict.keys(), out_list))
                validation_loss = outputs['validation_loss'][0] if type(outputs['validation_loss']) == list else outputs['validation_loss']
                validation_losses.append(validation_loss)
            except tf.errors.OutOfRangeError:
                break
            eval_counter += 1

        return np.mean(validation_losses)
    
    def _create_graphs(self):
        if 'train' in self._required_graphs:
            self._train_graph = tf.Graph()
            self._valLoss_graph = tf.Graph()
        if 'eval' in self._required_graphs:
            self._evaluate_graph = tf.Graph()
            if 'evaluateTrain' in self._hparams.modes:
                self._evaluateTrain_graph = tf.Graph()
            if 'evaluateAllTrainData' in self._hparams.modes:
                self._evaluateAllTrainData_graph = tf.Graph()
            #self._evaluateAudio_graph = tf.Graph()
            #self._evaluateVideo_graph = tf.Graph()
            #self._evaluateNoData_graph = tf.Graph()
        # self._predict_graph = tf.Graph()

    def _create_models(self):
        if 'train' in self._required_graphs:
            self._train_model = self._make_model(
                graph=self._train_graph,
                mode='train',
                batch_size=self._hparams.batch_size[0])
            self._valLoss_model = self._make_model(
                graph=self._valLoss_graph,
                mode='valLoss',
                batch_size=self._hparams.batch_size[0])

        if 'eval' in self._required_graphs:
            self._evaluate_model = self._make_model(
                graph=self._evaluate_graph,
                mode='evaluate',
                batch_size=self._hparams.batch_size[1])
            if 'evaluateTrain' in self._hparams.modes:
                self._evaluateTrain_model = self._make_model(
                    graph=self._evaluateTrain_graph,
                    mode='evaluateTrain',
                    batch_size=self._hparams.batch_size[1])
            if 'evaluateAllTrainData' in self._hparams.modes:
                self._evaluateAllTrainData_model = self._make_model(
                    graph=self._evaluateAllTrainData_graph,
                    mode='evaluateAllTrainData',
                    batch_size=self._hparams.batch_size[1])
            #self._evaluateAudio_model = self._make_model(
            #    graph=self._evaluateAudio_graph,
            #    mode='evaluateAudio',
            #    batch_size=self._hparams.batch_size[1])
            #self._evaluateVideo_model = self._make_model(
            #    graph=self._evaluateVideo_graph,
            #    mode='evaluateVideo',
            #    batch_size=self._hparams.batch_size[1])
            #self._evaluateNoData_model = self._make_model(
            #    graph=self._evaluateNoData_graph,
            #    mode='evaluateNoData',
            #    batch_size=self._hparams.batch_size[1])

    def _create_sessions(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        if 'train' in self._required_graphs:
            self._train_session = tf.Session(graph=self._train_graph, config=config)
            self._valLoss_session = tf.Session(graph=self._valLoss_graph, config=config)
        if 'eval' in self._required_graphs:
            self._evaluate_session = tf.Session(graph=self._evaluate_graph, config=config)
            if 'evaluateTrain' in self._hparams.modes:
                self._evaluateTrain_session = tf.Session(graph=self._evaluateTrain_graph, config=config)
            if 'evaluateAllTrainData' in self._hparams.modes:
                self._evaluateAllTrainData_session = tf.Session(graph=self._evaluateAllTrainData_graph, config=config)
            #self._evaluateAudio_session = tf.Session(graph=self._evaluateAudio_graph, config=config)
            #self._evaluateVideo_session = tf.Session(graph=self._evaluateVideo_graph, config=config)
            #self._evaluateNoData_session = tf.Session(graph=self._evaluateNoData_graph, config=config)
        # self._predict_session = tf.Session(graph=self._predict_graph, config=config)

        if self._hparams.profiling is True:
            from tensorflow.profiler import Profiler
            self.profiler = Profiler(self._train_session.graph)
            self.run_meta = tf.RunMetadata()
            makedirs('/tmp/timelines/', exist_ok=True)
            self.sess_opts = {
                'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                'run_metadata': self.run_meta
            }
        else:
            self.sess_opts = {}

    def _initialize_sessions(self):
        if 'train' in self._required_graphs:
            self._train_session.run(self._train_model.initializer)
            self._valLoss_session.run(self._valLoss_model.initializer)
        if 'eval' in self._required_graphs:
            self._evaluate_session.run(self._evaluate_model.initializer)
            if 'evaluateTrain' in self._hparams.modes:
                self._evaluateTrain_session.run(self._evaluateTrain_model.initializer)
            if 'evaluateAllTrainData' in self._hparams.modes:
                self._evaluateAllTrainData_session.run(self._evaluateAllTrainData_model.initializer)
            #self._evaluateAudio_session.run(self._evaluateAudio_model.initializer)
            #self._evaluateVideo_session.run(self._evaluateVideo_model.initializer)
            #self._evaluateNoData_session.run(self._evaluateNoData_model.initializer)

    def _make_model(self, graph, mode, batch_size):
        with graph.as_default():
            print('make_model', mode, batch_size)
            video_data, audio_data = self._fetch_data(mode, batch_size)
            print('data was fetched', mode)
            video_features, audio_features = self._preprocess_data(video_data, audio_data, mode, batch_size)
            
            #if mode == 'evaluateVideo' or mode == 'evaluateNoData':
            #    audio_features = audio_features._replace(inputs=tf.zeros(shape=tf.shape(audio_features.inputs), dtype=self._hparams.dtype))
            #if mode == 'evaluateAudio' or mode == 'evaluateNoData':
            #    video_features = video_features._replace(inputs=tf.zeros(shape=tf.shape(video_features.inputs), dtype=self._hparams.dtype))
            if mode == 'valLoss':
                mode = 'train'
            model = Seq2SeqModel(
                data_sequences=(video_features, audio_features),
                mode=mode,
                hparams=self._hparams
            )

            initializer = tf.global_variables_initializer()

            # Returning the original data, not the processed features
            return Model(data=(video_data, audio_data),
                         model=model,
                         initializer=initializer,
                         batch_size=batch_size)

    def _parse_iterator(self, iterator):
        inputs = tf.cast(iterator.inputs, dtype=self._hparams.dtype)
        labels = tf.cast(iterator.labels, tf.int32, name='labels')
        inputs_length = tf.cast(iterator.inputs_length, tf.int32, name='inputs_len')
        labels_length = tf.cast(iterator.labels_length, tf.int32, name='labels_len')

        return BatchedData(
            inputs=inputs,
            inputs_length=inputs_length,
            inputs_filenames=iterator.inputs_filenames,
            labels=labels,
            labels_length=labels_length,
            labels_filenames=iterator.labels_filenames,
            iterator_initializer=iterator.iterator_initializer,
            payload=None)

    def _parse_multimodal_iterator(self, iterator):
        vid_inputs = tf.cast(iterator.inputs[0], dtype=self._hparams.dtype, name='vid_inputs')
        vid_inputs_length = tf.cast(iterator.inputs_length[0], tf.int32, name='vid_inputs_len')
        aud_inputs = tf.cast(iterator.inputs[1], dtype=self._hparams.dtype, name='aud_inputs')
        aud_inputs_length = tf.cast(iterator.inputs_length[1], tf.int32, name='aud_inputs_len')

        labels = tf.cast(iterator.labels, tf.int32, name='labels')
        labels_length = tf.cast(iterator.labels_length, tf.int32, name='labels_len')

        if iterator.payload.get('aus', None) is not None:
            iterator.payload['aus'] = tf.cast(iterator.payload['aus'], dtype=self._hparams.dtype, name='action_units')

        video_data = BatchedData(
            inputs=vid_inputs,
            inputs_length=vid_inputs_length,
            inputs_filenames=iterator.inputs_filenames[0],
            labels=labels,
            labels_length=labels_length,
            labels_filenames=iterator.labels_filenames,
            iterator_initializer=iterator.iterator_initializer,
            payload=iterator.payload)

        audio_data = BatchedData(
            inputs=aud_inputs,
            inputs_length=aud_inputs_length,
            inputs_filenames=iterator.inputs_filenames[1],
            labels=labels,
            labels_length=labels_length,
            labels_filenames=iterator.labels_filenames,
            iterator_initializer=iterator.iterator_initializer,
            payload=None)  # currently we don't have any audio payload

        return video_data, audio_data

    def _fetch_data(self, mode, batch_size):

        video_data = None
        audio_data = None

        if self._video_processing is not None and self._audio_processing is not None:
            iterator = make_iterator_from_two_records(
                video_record=self._video_train_record if mode in ['train', 'evaluateAllTrainData'] else (
                    self._video_trainTest_record if mode == 'evaluateTrain' else self._video_test_record),
                audio_record=self._audio_train_record if mode in ['train', 'evaluateAllTrainData'] else (
                    self._audio_trainTest_record if mode == 'evaluateTrain' else self._audio_test_record),
                label_record=self._labels_train_record if mode in ['train', 'evaluateAllTrainData'] else (
                    self._labels_trainTest_record if mode == 'evaluateTrain' else self._labels_test_record),
                batch_size=batch_size,
                unit_dict=self._hparams.unit_dict,
                shuffle=True if mode == 'train' else False,
                reverse_input=False,
                bucket_width=45,  # video frame rate is 30 fps
            )
            video_data, audio_data = self._parse_multimodal_iterator(iterator)

        else:
            if self._video_processing is not None:
                video_iterator = make_iterator_from_one_record(
                    data_record=self._video_train_record if mode == 'train' else self._video_test_record,
                    label_record=self._labels_train_record if mode == 'train' else self._labels_test_record,
                    batch_size=batch_size,
                    unit_dict=self._hparams.unit_dict,
                    shuffle=True if mode == 'train' else False,
                    reverse_input=False,
                    bucket_width=45,  # video frame rate is 30 fps
                )

                video_data = self._parse_iterator(video_iterator)

            if self._audio_processing is not None:
                audio_iterator = make_iterator_from_one_record(
                    data_record=self._audio_train_record if mode == 'train' else self._audio_test_record,
                    label_record=self._labels_train_record if mode == 'train' else self._labels_test_record,
                    batch_size=batch_size,
                    shuffle=True if mode == 'train' else False,
                    unit_dict=self._hparams.unit_dict,
                    reverse_input=False,
                    bucket_width=45,  # audio feature rate is 30 mfcc/sec
                    max_sentence_length=self._hparams.max_sentence_length,
                )

                audio_data = self._parse_iterator(audio_iterator)
        print('fetch_data_mode', mode)
        print('fetch_data_video_data', video_data)
        print('fetch_data_audio_data', audio_data)
        if mode == 'evaluateVideo' or mode == 'evaluateNoData' or self._hparams.set_data_null == 'audio':
            #audio_data = audio_data._replace(inputs=tf.zeros_like(audio_data.inputs))
            audio_data = audio_data._replace(inputs=tf.zeros(shape=tf.shape(audio_data.inputs), dtype=self._hparams.dtype))
            print('fetch_data_audio_data', audio_data)
        if mode == 'evaluateAudio' or mode == 'evaluateNoData' or self._hparams.set_data_null == 'video':
            #video_data = video_data._replace(inputs=tf.zeros_like(video_data.inputs))
            video_data = video_data._replace(inputs=tf.zeros(shape=tf.shape(video_data.inputs), dtype=self._hparams.dtype))
            print('fetch_data_video_data', video_data)
        return video_data, audio_data

    def _preprocess_data(self, video_data, audio_data, mode, batch_size):

        if self._video_processing is not None:

            if 'cnn' in self._video_processing:

                with tf.variable_scope('CNN'):

                    visual_features = cnn_layers(
                        inputs=video_data.inputs,
                        cnn_type=self._video_processing,
                        is_training=(mode=='train'),
                        cnn_filters=self._hparams_video.cnn_filters,
                        cnn_dense_units=self._hparams_video.cnn_dense_units
                    )

                    # re-create video_data to update the `inputs` field
                    video_data = BatchedData(
                        inputs=visual_features,
                        inputs_length=video_data.inputs_length,
                        inputs_filenames=video_data.inputs_filenames,
                        labels=video_data.labels,
                        labels_length=video_data.labels_length,
                        labels_filenames=video_data.labels_filenames,
                        iterator_initializer=video_data.iterator_initializer,
                        payload=video_data.payload,
                    )

            elif self._video_processing == 'features':
                pass
            else:
                raise Exception('unknown visual content')
        else:
            pass

        if self._audio_processing is not None:

            if self._audio_processing == 'wav':  # compute mfcc on the fly

                audio_features = process_audio(
                    audio_data.inputs,
                    hparams=self._hparams_audio,
                )

                # re-create audio_data to update the `inputs` field
                audio_data = BatchedData(
                    inputs=audio_features,
                    inputs_length=audio_data.inputs_length,
                    inputs_filenames=audio_data.inputs_filenames,
                    labels=audio_data.labels,
                    labels_length=audio_data.labels_length,
                    labels_filenames=audio_data.labels_filenames,
                    iterator_initializer=audio_data.iterator_initializer,
                    payload=audio_data.payload,
                )

            elif self._audio_processing == 'features':
                pass
            else:
                raise Exception('unknown audio content')
        else:
            pass
        
        return video_data, audio_data

    def _initialize_summaries(self, summaries_dir, log_file_name):
        train_path = path.join(summaries_dir, "train", log_file_name)
        makedirs(path.dirname(train_path), exist_ok=True)
        self._train_summary_writer = tf.summary.FileWriter(train_path, self._train_graph, flush_secs=10)

        with self._train_graph.as_default():
            self._merged_train_summaries = tf.summary.merge_all()
    