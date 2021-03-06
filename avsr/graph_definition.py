"""
Graph creation functions.
"""


from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from avsr.basic_rnn_cells import BasicLSTMCell, BasicGRUCell
from avsr.skip_rnn_cells import SkipLSTMCell, MultiSkipLSTMCell
from avsr.skip_rnn_cells import SkipGRUCell, MultiSkipGRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from avsr.own_cells import SkipDropoutWrapper
from tensorflow.contrib.rnn import LSTMCell

def create_generic_flags():
    """
    Create flags which are shared by all experiments
    """
    # Generic flags
    tf.app.flags.DEFINE_string('model', 'lstm', "Select RNN cell: {lstm, gru, skip_lstm, skip_gru}")
    tf.app.flags.DEFINE_integer("rnn_cells", 110, "Number of RNN cells.")
    tf.app.flags.DEFINE_integer("rnn_layers", 1, "Number of RNN layers.")
    tf.app.flags.DEFINE_integer('batch_size', 256, "Batch size.")
    tf.app.flags.DEFINE_float('learning_rate', 0.0001, "Learning rate.")
    tf.app.flags.DEFINE_float('grad_clip', 1., "Clip gradients at this value. Set to <=0 to disable clipping.")
    tf.app.flags.DEFINE_string('logdir', '../logs', "Directory where TensorBoard logs will be stored.")

    # Flags for the Skip RNN cells
    tf.app.flags.DEFINE_float('cost_per_sample', 0., "Cost per used sample. Set to 0 to disable this option.")


def compute_gradients(loss, learning_rate, gradient_clipping=-1):
    """
    Create optimizer, compute gradients and (optionally) apply gradient clipping
    """
    opt = tf.train.AdamOptimizer(learning_rate)
    if gradient_clipping > 0:
        vars_to_optimize = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, vars_to_optimize), clip_norm=gradient_clipping)
        grads_and_vars = list(zip(grads, vars_to_optimize))
    else:
        grads_and_vars = opt.compute_gradients(loss)
    return opt, grads_and_vars


def create_model(model, num_cells, batch_size, as_list=False, learn_initial_state=True, use_dropout=False, dropout_probability=(1, 1, 1)):
    """
    Returns a tuple of (cell, initial_state) to use with dynamic_rnn.
    If num_cells is an integer, a single RNN cell will be created. If it is a list, a stack of len(num_cells)
    cells will be created.
    """
    if not model in ['lstm', 'gru', 'skip_lstm', 'skip_gru', 'multi_skip_lstm']:
        raise ValueError('The specified model is not supported. Please use {lstm, gru, skip_lstm, skip_gru}.')
    if (isinstance(num_cells, list) or isinstance(num_cells, tuple)) and len(num_cells) > 1:
        if model == 'skip_lstm':
            if as_list:
                cells = []
                for i, n in enumerate(num_cells):
                    if i == len(num_cells)-1:
                        cell = SkipLSTMCell(n)
                    else:
                        cell = BasicLSTMCell(num_units=n,
                                             use_peepholes=False,
                                             cell_clip=1.0,
                                             initializer=tf.variance_scaling_initializer(),
                                             dtype=tf.float32)
                    cell.layer = i
                    cells.append(cell)
        elif model == 'multi_skip_lstm':
            cells = MultiSkipLSTMCell(num_cells)
        elif model == 'skip_gru':
            if as_list:
                cells = []
                for i, n in enumerate(num_cells):
                    cell = SkipGRUCell(n)
                    cell.layer = i
                    cells.append(cell)
            else:
                cells = MultiSkipGRUCell(num_cells)
        elif model == 'lstm':
            cell_list = [BasicLSTMCell(num_units=n,
                                     use_peepholes=False,
                                     cell_clip=1.0,
                                     initializer=tf.variance_scaling_initializer(),
                                     dtype=tf.float32) for n in num_cells]
            if as_list:
                cells = cell_list
            else:
                cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        elif model == 'gru':
            cell_list = [BasicGRUCell(n) for n in num_cells]
            if as_list:
                cells = cell_list
            else:
                cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        if learn_initial_state:
            print('learn_initial_state_model', model)
            if model == 'skip_lstm' or model == 'skip_gru':
                if as_list:
                    initial_state = []
                    for idx, cell in enumerate(cells):
                        with tf.variable_scope('layer_%d' % (idx + 1)):
                            initial_state.append(cell.trainable_initial_state(batch_size))
                    initial_state = tuple(initial_state)
                else:
                    initial_state = cells.trainable_initial_state(batch_size)
            elif model == 'multi_skip_lstm':
                initial_state = cells.trainable_initial_state(batch_size)
                initial_state = tuple(initial_state)
            else:
                initial_state = []
                for idx, cell in enumerate(cell_list):
                    #with tf.variable_scope('layer_%d' % (idx + 1)):
                    initial_state.append(cell.trainable_initial_state(batch_size))
                initial_state = tuple(initial_state)
        else:
            initial_state = None
        print('create_model_cells', cells)
        if use_dropout:
            if isinstance(cells, list) or isinstance(cells, tuple):
                drop_cells = []
                for cell in cells:
                    if isinstance(cell, SkipLSTMCell):
                        drop_cell = SkipDropoutWrapper(cell,
                                                   input_keep_prob=dropout_probability[0],
                                                   state_keep_prob=dropout_probability[1],
                                                   output_keep_prob=dropout_probability[2],
                                                   variational_recurrent=False,
                                                   dtype=tf.float32,
                                                   # input_size=self._inputs.get_shape()[1:],
                                                   )
                    else:
                        drop_cell = DropoutWrapper(cell,
                                                       input_keep_prob=dropout_probability[0],
                                                       state_keep_prob=dropout_probability[1],
                                                       output_keep_prob=dropout_probability[2],
                                                       variational_recurrent=False,
                                                       dtype=tf.float32,
                                                       # input_size=self._inputs.get_shape()[1:],
                                                       )
                    drop_cells.append(drop_cell)
                cells = drop_cells
            else:
                cells = SkipDropoutWrapper(cells,
                                               input_keep_prob=dropout_probability[0],
                                               state_keep_prob=dropout_probability[1],
                                               output_keep_prob=dropout_probability[2],
                                               variational_recurrent=False,
                                               dtype=tf.float32,
                                               # input_size=self._inputs.get_shape()[1:],
                                               )
        return cells, initial_state
    else:
        if isinstance(num_cells, list) or isinstance(num_cells, tuple):
            num_cells = num_cells[0]
        if model == 'skip_lstm':
            cell = SkipLSTMCell(num_cells)
        elif model == 'skip_gru':
            cell = SkipGRUCell(num_cells)
        elif model == 'lstm':
            cell = BasicLSTMCell(num_units=num_cells,
                                 use_peepholes=False,
                                 cell_clip=1.0,
                                 initializer=tf.variance_scaling_initializer(),
                                 dtype=tf.float32)
        elif model == 'gru':
            cell = BasicGRUCell(num_cells)
        if learn_initial_state:
            initial_state = cell.trainable_initial_state(batch_size)
        else:
            initial_state = None
        return cell, initial_state


def using_skip_rnn(model):
    """
    Helper function determining whether a Skip RNN models is being used
    """
    return model.lower() == 'skip_lstm' or model.lower() == 'skip_gru'


def split_rnn_outputs(model, rnn_outputs):
    """
    Split the output of dynamic_rnn into the actual RNN outputs and the state update gate
    """
    if using_skip_rnn(model):
        return rnn_outputs.h, rnn_outputs.state_gate
    else:
        return rnn_outputs, tf.no_op()


def compute_budget_loss(model, loss, updated_states, cost_per_sample):
    """
    Compute penalization term on the number of updated states (i.e. used samples)
    """
    if using_skip_rnn(model):
        return tf.reduce_mean(tf.reduce_sum(cost_per_sample * updated_states, 1), 0)
    else:
        return tf.zeros(loss.get_shape())
