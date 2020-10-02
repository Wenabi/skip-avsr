import tensorflow as tf
import collections
from .cells import build_rnn_layers
from tensorflow.contrib.rnn import MultiRNNCell
from .attention import add_attention

from tensorflow.python.layers.core import Dense
from tensorflow.contrib import seq2seq
from avsr.skip_rnn_cells import SkipLSTMStateTuple, SkipLSTMCell
from avsr.own_cells import SkipAttentionWrapper, SkipMultiRNNCell
from avsr.graph_definition import create_model
from tensorflow.contrib.rnn import LSTMStateTuple

SkipInfoTuple = collections.namedtuple("SkipInfoTuple", ("updated_states", "meanUpdates", "budget_loss"))

class EncoderData(collections.namedtuple("EncoderData", ("outputs", "final_state"))):
    pass


class Seq2SeqEncoder(object):

    def __init__(self,
                 data,
                 mode,
                 hparams,
                 num_units_per_layer,
                 dropout_probability,
                 data_type,
                 **kwargs
                 ):

        self._data = data
        self._mode = mode
        self._hparams = hparams
        self._num_units_per_layer = num_units_per_layer
        self._dropout_probability = dropout_probability
        self._data_type = data_type
        self.skip_infos = None

        self._init_data()
        self._init_encoder()

        if kwargs.get('regress_aus', False) and mode == 'train':
            self._init_au_loss()

    def _init_data(self):
        self._inputs = self._data.inputs
        self._inputs_len = self._data.inputs_length

        # self._labels = self._data.labels
        # self._labels_len = self._data.labels_length

        if self._hparams.batch_normalisation is True:
            self._inputs = tf.layers.batch_normalization(
                inputs=self._inputs,
                axis=-1,
                training=(self._mode == 'train'),
                fused=True,
            )
        if self._hparams.instance_normalisation is True:
            from tensorflow.contrib.layers import instance_norm
            self._inputs = instance_norm(
                inputs=self._inputs,
            )

    def _init_encoder(self):
        r"""
        Instantiates the seq2seq encoder
        :return:
        """
        with tf.variable_scope("Encoder") as scope:

            encoder_inputs = self._maybe_add_dense_layers()
            # encoder_inputs = a_resnet(encoder_inputs, self._mode == 'train')

            if self._hparams.encoder_type == 'unidirectional':
                cell_type = self._hparams.cell_type[0 if self._data_type == 'video' else 1]
                batch_size = self._hparams.batch_size[0 if self._mode == 'train' else 1]
                self._encoder_cells, initial_state = build_rnn_layers(
                    cell_type=cell_type,
                    num_units_per_layer=self._num_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._dropout_probability,
                    mode=self._mode,
                    residual_connections=self._hparams.residual_encoder,
                    highway_connections=self._hparams.highway_encoder,
                    batch_size=batch_size,
                    dtype=self._hparams.dtype,
                    weight_sharing=self._hparams.encoder_weight_sharing,
                    as_list=True
                )

                #self._encoder_cells, initial_states = create_model(model=cell_type,
                #                                                   num_cells=self._num_units_per_layer,
                #                                                   batch_size=batch_size,
                #                                                   as_list=True,
                #                                                   learn_initial_state=True if 'skip' in cell_type else False,
                #                                                   use_dropout=self._hparams.use_dropout,
                #                                                   dropout_probability=self._dropout_probability)
                
                print("Seq2Seq_encoder_cells", self._encoder_cells)
                print("Seq2Seq_initial_state", initial_state)
                print("Seq2Seq_encoder_encoder_inputs", encoder_inputs)
                #if "skip" in cell_type:
                #    if isinstance(self._encoder_cells, list):
                #        initial_state = self._encoder_cells[0].trainable_initial_state(batch_size)
                #    else:
                #        initial_state = self._encoder_cells.trainable_initial_state(batch_size)
                #        self._encoder_cells = [self._encoder_cells]
                #    print('Seq2Seq_initial_state', initial_state)
                #    print('Seq2Seq_scope', scope)
                #    out = tf.nn.dynamic_rnn(
                #        cell=MultiRNNCell(self._encoder_cells),
                #        inputs=encoder_inputs,
                #        sequence_length=self._inputs_len,
                #        parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                #        swap_memory=False,
                #        dtype=self._hparams.dtype,
                #        scope=scope,
                #        initial_state=(initial_state,),
                #    )
                #else:
                self._encoder_cells = maybe_list(self._encoder_cells)
                self._encoder_cells = maybe_multirnn(self._encoder_cells)
                initial_state = initial_state if len(self._num_units_per_layer) > 1 else initial_state[0]
                if cell_type == 'skip_lstm':
                    out = tf.nn.dynamic_rnn(
                        cell=self._encoder_cells,
                        inputs=encoder_inputs,
                        sequence_length=self._inputs_len,
                        parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                        swap_memory=False,
                        dtype=self._hparams.dtype,
                        initial_state=initial_state,
                        scope=scope,
                    )
                else:
                    out = tf.nn.dynamic_rnn(
                        cell=self._encoder_cells,
                        inputs=encoder_inputs,
                        sequence_length=self._inputs_len,
                        parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                        swap_memory=False,
                        dtype=self._hparams.dtype,
                        initial_state=initial_state,
                        scope=scope,
                    )

                print("Seq2SeqEncoder_dynamic_rnn_out", out)
                self._encoder_outputs, self._encoder_final_state = out
                if "skip" in cell_type:
                    self._encoder_outputs, updated_states = self._encoder_outputs
                    print("Seq2SeqEncoder_updated_states", updated_states)
                    cost_per_sample = self._hparams.cost_per_sample[0] if self._data_type == 'video' \
                        else self._hparams.cost_per_sample[1]
                    budget_loss = tf.reduce_mean(tf.reduce_sum(cost_per_sample * updated_states, 1), 0)
                    meanUpdates = tf.reduce_mean(tf.reduce_sum(updated_states, 1), 0)
                    self.skip_infos = SkipInfoTuple(updated_states, meanUpdates, budget_loss)

            elif self._hparams.encoder_type == 'bidirectional':

                self._fw_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type[0 if self._data_type == 'video' else 1],
                    num_units_per_layer=self._num_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._dropout_probability,
                    mode=self._mode,
                    dtype=self._hparams.dtype,
                )

                self._bw_cells = build_rnn_layers(
                    cell_type=self._hparams.cell_type[0 if self._data_type == 'video' else 1],
                    num_units_per_layer=self._num_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._dropout_probability,
                    mode=self._mode,
                    dtype=self._hparams.dtype,
                )

                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self._fw_cells,
                    cell_bw=self._bw_cells,
                    inputs=encoder_inputs,
                    sequence_length=self._inputs_len,
                    dtype=self._hparams.dtype,
                    parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                    swap_memory=False,
                    scope=scope,
                )

                self._encoder_outputs = tf.concat(bi_outputs, -1)
                encoder_state = []

                for layer in range(len(bi_state[0])):
                    fw_state = bi_state[0][layer]
                    bw_state = bi_state[1][layer]
                    cell_type = self._hparams.cell_type[0 if self._data_type == 'video' else 1]
                    if cell_type == 'gru':
                        cat = tf.concat([fw_state, bw_state], axis=-1)
                        proj = tf.layers.dense(cat, units=self._hparams.decoder_units_per_layer[0], use_bias=False)
                        encoder_state.append(proj)
                    elif cell_type == 'lstm':
                        cat_c = tf.concat([fw_state.c, bw_state.c], axis=-1)
                        cat_h = tf.concat([fw_state.h, bw_state.h], axis=-1)
                        proj_c = tf.layers.dense(cat_c, units=self._hparams.decoder_units_per_layer[0], use_bias=False)
                        proj_h = tf.layers.dense(cat_h, units=self._hparams.decoder_units_per_layer[0], use_bias=False)
                        state_tuple = tf.contrib.rnn.LSTMStateTuple(c=proj_c, h=proj_h)
                        encoder_state.append(state_tuple)
                    else:
                        raise ValueError('BiRNN fusion strategy not implemented for this cell')
                encoder_state = tuple(encoder_state)

                self._encoder_final_state = encoder_state

            else:
                raise Exception('Allowed encoder types: `unidirectional`, `bidirectional`')

    def _maybe_add_dense_layers(self):
        r"""
        Optionally passes self._input through several Fully Connected (Dense) layers
        with the configuration defined by the self._input_dense_layers tuple

        Returns
        -------
        The output of the network of Dense layers
        """
        layer_inputs = self._inputs
        if self._hparams.input_dense_layers[0] > 0:

            fc = [Dense(units,
                        activation=tf.nn.selu,
                        use_bias=False,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
                  for units in self._hparams.input_dense_layers]

            for layer in fc:
                layer_inputs = layer(layer_inputs)
        else:
            pass
        return layer_inputs

    def _init_au_loss(self):
        encoder_output_layer = tf.layers.Dense(
            units=2, activation=tf.nn.sigmoid,
            )

        projected_outputs = encoder_output_layer(self._encoder_outputs)
        normed_aus = tf.clip_by_value(self._data.payload['aus'], 0.0, 3.0) / 3.0

        mask = tf.sequence_mask(self._inputs_len, dtype=self._hparams.dtype)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 2])

        self.au_loss = tf.losses.mean_squared_error(
            predictions=projected_outputs,
            labels=normed_aus,
            weights=mask
        )

    def get_data(self):

        return EncoderData(
            outputs=self._encoder_outputs,
            final_state=self._encoder_final_state
        )


class AttentiveEncoder(Seq2SeqEncoder):

    def __init__(self,
                 data,
                 mode,
                 hparams,
                 num_units_per_layer,
                 attended_memory,
                 attended_memory_length,
                 dropout_probability,
                 data_type):
        r"""
        Implements https://arxiv.org/abs/1809.01728
        """

        self._attended_memory = attended_memory
        self._attended_memory_length = attended_memory_length
        self._data_type = data_type
        self._skip_infos = None

        super(AttentiveEncoder, self).__init__(
            data,
            mode,
            hparams,
            num_units_per_layer,
            dropout_probability,
            data_type
        )

    def _init_encoder(self):
        with tf.variable_scope("Encoder") as scope:

            encoder_inputs = self._maybe_add_dense_layers()
            cell_type = self._hparams.cell_type[0 if self._data_type == 'video' else 1]
            batch_size = self._hparams.batch_size[0 if self._mode == 'train' else 1]
            if self._hparams.encoder_type == 'unidirectional':
                self._encoder_cells, initial_state = build_rnn_layers(
                    cell_type=cell_type,
                    num_units_per_layer=self._num_units_per_layer,
                    use_dropout=self._hparams.use_dropout,
                    dropout_probability=self._dropout_probability,
                    mode=self._mode,
                    as_list=True,
                    batch_size=batch_size,
                    dtype=self._hparams.dtype)

                #self._encoder_cells, initial_state = create_model(model=cell_type,
                #                                     num_cells=self._num_units_per_layer,
                #                                     batch_size=batch_size,
                #                                     as_list=False if cell_type == 'multi_skip_lstm' else True,
                #                                     learn_initial_state=True if 'skip' in cell_type else False,
                #                                     use_dropout=self._hparams.use_dropout,
                #                                     dropout_probability=self._dropout_probability)
                
                self._encoder_cells = maybe_list(self._encoder_cells)
                print(self._num_units_per_layer)
                print('encoder_cells', self._encoder_cells)
                print('AttentiveEncoder_initial_state', initial_state)

                #### here weird code

                # 1. reverse mem
                # self._attended_memory = tf.reverse(self._attended_memory, axis=[1])

                # 2. append zeros
                # randval1 = tf.random.uniform(shape=[], minval=25, maxval=100, dtype=tf.int32)
                # randval2 = tf.random.uniform(shape=[], minval=25, maxval=100, dtype=tf.int32)
                # zeros_slice1 = tf.zeros([1, randval1, 256], dtype=tf.float32)  # assuming we use inference on a batch size of 1
                # zeros_slice2 = tf.zeros([1, randval2, 256], dtype=tf.float32)
                # self._attended_memory = tf.concat([zeros_slice1, self._attended_memory, zeros_slice2], axis=1)
                # self._attended_memory_length += randval1 + randval2

                # 3. blank mem
                # self._attended_memory = 0* self._attended_memory

                # 4. mix with noise
                # noise = tf.random.truncated_normal(shape=tf.shape(self._attended_memory))
                # noise = tf.random.uniform(shape=tf.shape(self._attended_memory))

                # self._attended_memory = noise

                #### here stop weird code
                if cell_type == 'skip_lstm' and self._hparams.separate_skip_rnn:
                    skip_cell = self._encoder_cells[0]
                    self._encoder_cells = self._encoder_cells[1:]
                    
                    skip_out = tf.nn.dynamic_rnn(
                        cell=skip_cell,
                        inputs=encoder_inputs,
                        sequence_length=self._inputs_len,
                        parallel_iterations=batch_size,
                        swap_memory=False,
                        dtype=self._hparams.dtype,
                        scope=scope,
                        initial_state=skip_cell.trainable_initial_state(batch_size),
                    )
                    print('skip_out', skip_out)
                    skip_output, skip_final_state = skip_out
                    h, updated_states = skip_output
                    print('skip_encoder_updated_states', updated_states)
                    cost_per_sample = self._hparams.cost_per_sample[1]
                    budget_loss = tf.reduce_mean(tf.reduce_sum(cost_per_sample * updated_states, 1), 0)
                    meanUpdates = tf.reduce_mean(tf.reduce_sum(updated_states, 1), 0)
                    self.skip_infos = SkipInfoTuple(updated_states, meanUpdates, budget_loss)
                    # Tried to remove the skipped states in the output, but this destroys the input shape
                    #updated_states_shape = tf.shape(updated_states)
                    #h_shape = tf.shape(h)
                    #updated_states = tf.reshape(updated_states, [batch_size, updated_states_shape[1]])
                    #new_h = tf.boolean_mask(h, tf.where(updated_states == 1.))
                    #new_h = tf.where(updated_states == 1., h, tf.zeros(shape=h_shape))
                    #print('new_h', new_h)
                    #new_h_shape = tf.shape(new_h)
                    #new_h = tf.reshape(new_h, [batch_size, new_h_shape[0] / batch_size, new_h_shape[1]])
                    #self._inputs_len = (batch_size, new_h_shape[0] / batch_size)
                
                    print('skip_encoder_layer_h', h)
                print('attended_memory', self._attended_memory)
                print(self._encoder_cells)
                attention_cells, dummy_initial_state = add_attention(
                    cell_type='lstm' if self._hparams.separate_skip_rnn else cell_type,
                    cells=self._encoder_cells[-1],
                    attention_types=self._hparams.attention_type[0],
                    num_units=self._num_units_per_layer[-1],
                    memory=self._attended_memory,
                    memory_len=self._attended_memory_length,
                    mode=self._mode,
                    dtype=self._hparams.dtype,
                    initial_state=initial_state[-1] if (isinstance(initial_state, tuple) and not isinstance(initial_state, SkipLSTMStateTuple)) else initial_state,
                    batch_size=tf.shape(self._inputs_len),
                    write_attention_alignment=self._hparams.write_attention_alignment,
                    fusion_type='linear_fusion',
                )
                print('AttentiveEncoder_initial_state2', initial_state)
                if isinstance(initial_state, tuple) and not isinstance(initial_state, SkipLSTMStateTuple):
                    initial_state = list(initial_state)
                    initial_state[-1] = dummy_initial_state
                    initial_state = tuple(initial_state)
                else:
                    initial_state = dummy_initial_state
                self._encoder_cells[-1] = attention_cells

                # initial_state = self._encoder_cells.get_initial_state(inputs=None, batch_size=batch_size, dtype=self._hparams.dtype)
                #initial_state = []
                #for i, cell in enumerate(self._encoder_cells):
                #    print(i, cell)
                #    if isinstance(cell, SkipLSTMCell):
                #        #pass
                #        #with tf.variable_scope(f'layer_{i}') as init_scope:
                #        initial_state.append(cell.zero_state(batch_size, dtype=self._hparams.dtype))
                #    else:
                #        with tf.variable_scope(f'layer_{i}'):
                #            initial_state.append(
                #                cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=self._hparams.dtype))
                #initial_state = tuple(initial_state)

                print('AttentiveEncoder_encoder_cells', self._encoder_cells)
                self._encoder_cells = maybe_multirnn(self._encoder_cells)
                
                print('AttentiveEncoder_encoder_cells', self._encoder_cells)
                print('AttentiveEncoder_encoder_inputs', encoder_inputs)
                print('AttentiveEncoder_inputs_len', self._inputs_len)
                #initial_state = self._encoder_cells.get_initial_state(batch_size=batch_size, dtype=self._hparams.dtype)
                #print('AttentiveEncoder_initial_state_final', initial_state)
                out = tf.nn.dynamic_rnn(
                    cell=self._encoder_cells,
                    inputs=encoder_inputs if not self._hparams.separate_skip_rnn else h,
                    sequence_length=self._inputs_len,
                    parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                    swap_memory=False,
                    dtype=self._hparams.dtype,
                    scope=scope,
                    initial_state=initial_state,
                    )

                self._encoder_outputs, self._encoder_final_state = out
                print("AttentiveEncoder_dynamic_rnn_out", self._encoder_outputs)
                print("AttentiveEncoder_dynamic_rnn_fs", self._encoder_final_state)
                
                if not self._hparams.separate_skip_rnn and 'skip' in cell_type:
                    self._encoder_outputs, updated_states = self._encoder_outputs
                    cost_per_sample = self._hparams.cost_per_sample[1]
                    budget_loss = tf.reduce_mean(tf.reduce_sum(cost_per_sample * updated_states, 1), 0)
                    meanUpdates = tf.reduce_mean(tf.reduce_sum(updated_states, 1), 0)
                    self.skip_infos = SkipInfoTuple(updated_states, meanUpdates, budget_loss)
                    
                    if isinstance(self._encoder_final_state, tuple) and not isinstance(self._encoder_final_state, seq2seq.AttentionWrapperState):
                        self._encoder_final_state = self._encoder_final_state[-1]
                    print('AttentiveEncoder_final_state_inBetween', self._encoder_final_state)
                    if isinstance(self._encoder_final_state, seq2seq.AttentionWrapperState):
                        cell_state = self._encoder_final_state.cell_state
                        try:
                            cell_state = [LSTMStateTuple(cs.c, cs.h) for cs in cell_state]
                        except:
                            cell_state = LSTMStateTuple(cell_state.c, cell_state.h)
                        self._encoder_final_state = seq2seq.AttentionWrapperState(cell_state,
                                                                                  self._encoder_final_state.attention,
                                                                                  self._encoder_final_state.time,
                                                                                  self._encoder_final_state.alignments,
                                                                                  self._encoder_final_state.alignment_history,
                                                                                  self._encoder_final_state.attention_state)
                    print('AttentiveEncoder_final_state', self._encoder_final_state)
                    
                if self._hparams.write_attention_alignment is True:
                    # self.weights_summary = self._encoder_final_state[-1].attention_weight_history.stack()
                    self.attention_summary, self.attention_alignment = self._create_attention_alignments_summary(maybe_list(self._encoder_final_state)[-1])

    def _create_attention_alignments_summary(self, states):
        r"""
        Generates the alignment images, useful for visualisation/debugging purposes
        """
        attention_alignment = states.alignment_history[0].stack()

        attention_alignment = tf.expand_dims(tf.transpose(attention_alignment, [1, 2, 0]), -1)

        # attention_images_scaled = tf.image.resize_images(1-attention_images, (256,128))
        attention_images = 1 - attention_alignment

        attention_summary = tf.summary.image("attention_images_cm", attention_images,
                                             max_outputs=self._hparams.batch_size[1])

        return attention_summary, attention_alignment

    def get_data(self):

        def prepare_final_state(state):
            r"""
            state is a stack of zero or several RNN cells, followed by a final Attention wrapped RNN cell
            """

            from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionWrapperState
            print('prepare_final_state', state)
            final_state = []
            if type(state) is tuple:
                for cell in state:
                    if type(cell) == AttentionWrapperState:
                        final_state.append(cell.cell_state)
                    else:
                        print('prepare_final_state_cell', cell)
                        final_state.append(cell)
                print('prepare_final_State', final_state)
                return final_state
            else:  # only one RNN layer of attention wrapped cells
                print('prepare_final_state', state.cell_state)
                return state.cell_state

        return EncoderData(
            outputs=self._encoder_outputs,
            final_state=prepare_final_state(self._encoder_final_state)
        )


def maybe_list(obj):
    if type(obj) in (list, tuple):
        return obj
    else:
        return [obj, ]


def maybe_multirnn(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return MultiRNNCell(lst)
