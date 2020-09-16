import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, DeviceWrapper, DropoutWrapper, \
    LSTMCell, GRUCell, LSTMBlockCell, UGRNNCell, NASCell, GRUBlockCellV2, \
    HighwayWrapper, ResidualWrapper
from avsr.skip_rnn_cells import SkipLSTMCell
from avsr.own_cells import SkipDropoutWrapper, SkipMultiRNNCell
from avsr.basic_rnn_cells import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.rnn_cell import LayerNormLSTMCell, LayerNormBasicLSTMCell
def _build_single_cell(layer, cell_type, num_units, use_dropout, mode, dropout_probability, batch_size, dtype, device=None):
    r"""

    :param num_units: `int`
    :return:
    """
    print('cell_type', cell_type)
    initial_state = None
    if cell_type == 'skip_lstm':
        cells = SkipLSTMCell(num_units=num_units)
        cells.layer = layer
        initial_state = cells.trainable_initial_state(batch_size=batch_size)
    elif cell_type == 'lstm':
        cells = BasicLSTMCell(num_units=num_units,
                         use_peepholes=False,
                         cell_clip=1.0,
                         initializer=tf.variance_scaling_initializer(),
                         dtype=dtype)
        initial_state = cells.zero_state(batch_size, dtype)
    elif cell_type == 'layernorm_lstm':
        cells = LayerNormLSTMCell(num_units=num_units,
                                  cell_clip=1.0)
    elif cell_type == 'layernorm_basiclstm':
        cells = LayerNormBasicLSTMCell(num_units=num_units)
    elif cell_type == 'gru':
        cells = GRUCell(num_units=num_units,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        bias_initializer=tf.variance_scaling_initializer(),
                        dtype=dtype
                        )
    elif cell_type == 'ugrnn':
        cells = UGRNNCell(num_units)
    elif cell_type == 'lstm_block':
        cells = LSTMBlockCell(num_units=num_units,
                              use_peephole=True,
                              cell_clip=None)
    elif cell_type == 'gru_block':
        cells = GRUBlockCellV2(num_units=num_units)
    elif cell_type == 'nas':
        cells = NASCell(num_units=num_units)
    elif cell_type == 'lstm_masked':
        from tensorflow.contrib.model_pruning import MaskedLSTMCell
        cells = MaskedLSTMCell(num_units=num_units)
    else:
        raise Exception('cell type not supported: {}'.format(cell_type))
    print('build_single_cell_cells', cells)
    if use_dropout is True and mode == 'train':
        if 'skip' in cell_type:
            cells = SkipDropoutWrapper(cells,
                                       input_keep_prob=dropout_probability[0],
                                       state_keep_prob=dropout_probability[1],
                                       output_keep_prob=dropout_probability[2],
                                       variational_recurrent=False,
                                       dtype=dtype,
                                       # input_size=self._inputs.get_shape()[1:],
                                       )
        else:
            cells = DropoutWrapper(cells,
                                   input_keep_prob=dropout_probability[0],
                                   state_keep_prob=dropout_probability[1],
                                   output_keep_prob=dropout_probability[2],
                                   variational_recurrent=False,
                                   dtype=dtype,
                                   # input_size=self._inputs.get_shape()[1:],
                                   )
    if device is not None:
        cells = DeviceWrapper(cells, device=device)
    print('build_single_cell', layer)
    print('build_single_cell', cells)
    print('build_single_cell', initial_state)
    return cells, initial_state


def build_rnn_layers(
        cell_type,
        num_units_per_layer,
        use_dropout,
        dropout_probability,
        mode,
        batch_size,
        dtype,
        residual_connections=False,
        highway_connections=False,
        weight_sharing=False,
        as_list=False,
    ):

    cell_list = []
    initial_state_list = []
    for layer, units in enumerate(num_units_per_layer):
        #if layer > 0 and cell_type == 'skip_lstm':#CHANGED so only the first layer is a skip_lstm
        #    cell_type = 'lstm'
        print('build_rnn_layers', layer, cell_type)
        if layer > 1 and weight_sharing is True:
            cell = cell_list[-1]
        else:
            if cell_type == 'skip_lstm' and layer == len(num_units_per_layer)-1:
                single_cell_type = 'skip_lstm'
            else:
                single_cell_type = 'lstm'
            cell, initial_state = _build_single_cell(
                layer=layer,
                cell_type=single_cell_type,
                num_units=units,
                use_dropout=use_dropout,
                dropout_probability=dropout_probability,
                mode=mode,
                batch_size=batch_size,
                dtype=dtype,
            )

            if highway_connections is True and layer > 0:
                cell = HighwayWrapper(cell)
            elif residual_connections is True and layer > 0:
                cell = ResidualWrapper(cell)

        cell_list.append(cell)
        initial_state_list.append(initial_state)
    
    initial_state_list = tuple(initial_state_list)
    
    if len(cell_list) == 1:
        return cell_list[0], initial_state_list
    else:
        if as_list is False:
            return SkipMultiRNNCell(cell_list)
        else:
            return cell_list, initial_state_list
