import tensorflow as tf
import collections
from tensorflow.contrib.seq2seq import AttentionWrapperState, AttentionWrapper, AttentionMechanism
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from avsr.skip_rnn_cells import SkipLSTMStateTuple, SkipGRUStateTuple, SkipLSTMOutputTuple, SkipGRUOutputTuple, \
    SkipLSTMCell, SkipGRUCell, MultiSkipLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper, LSTMStateTuple, MultiRNNCell

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)
    
    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])
    
    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context
    
    return attention, alignments, next_attention_state


class SkipAttentionWrapper(AttentionWrapper):
    """Wraps another `RNNCell` with attention.
    """
    
    @property
    def output_size(self):
        if self._output_attention:
            return SkipLSTMOutputTuple(self._attention_layer_size, 1)
        else:
            return self._cell.output_size
    
    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.
        Returns:
          An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return AttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._item_or_tuple(
                a.alignments_size for a in self._attention_mechanisms),
            attention_state=self._item_or_tuple(
                a.state_size for a in self._attention_mechanisms),
            alignment_history=self._item_or_tuple(
                a.alignments_size if self._alignment_history else ()
                for a in self._attention_mechanisms))  # sometimes a TensorArray
    
    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.
        Returns:
          An `AttentionWrapperState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.
        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            print('SkipAttentionWrapper_cell', self._cell)
            print('SkipAttentionWrapper_initial_cell_state', self._initial_cell_state)
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            print('SkipAttentionWrapper_cell_state', cell_state)
            error_message = (
                    "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.  Are you using "
                    "the BeamSearchDecoder?  If so, make sure your encoder output has "
                    "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                    "the batch_size= argument passed to zero_state is "
                    "batch_size * beam_width.")
            with ops.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            initial_alignments = [
                attention_mechanism.initial_alignments(batch_size, dtype)
                for attention_mechanism in self._attention_mechanisms]
            return AttentionWrapperState(
                cell_state=cell_state,
                time=array_ops.zeros([], dtype=dtypes.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                              dtype),
                alignments=self._item_or_tuple(initial_alignments),
                attention_state=self._item_or_tuple(
                    attention_mechanism.initial_state(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                alignment_history=self._item_or_tuple(
                    tensor_array_ops.TensorArray(
                        dtype,
                        size=0,
                        dynamic_size=True,
                        element_shape=alignment.shape)
                    if self._alignment_history else ()
                    for alignment in initial_alignments))
    
    def call(self, inputs, state):
        print('SkipAttentionWrapper_inputs', inputs)
        print('SkipAttentionWrapper_state', state)
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))
        
        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        print('SkipAttentionWrapper_cell_inputs', cell_inputs)
        print('SkipAttentionWrapper_cell_state', cell_state)
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        
        print('SkipAttentionWrapper_cell_inputs', cell_inputs)
        print('SkipAttentionWrapper_cell_state', cell_state)
        print('SkipAttentionWrapper_cell_output', cell_output)
        print('SkipAttentionWrapper_next_cell_state', next_cell_state)
        
        SkipOutput = False
        if isinstance(cell_output, SkipLSTMOutputTuple):
            SkipOutput = True
            cell_output, state_gate = cell_output
        
        cell_batch_size = (
                tensor_shape.dimension_value(cell_output.shape[0]) or
                array_ops.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")
        
        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]
        
        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism, cell_output, previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()
            
            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)
        
        attention = array_ops.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))
        
        print('MyAttentionWrapper_attention', attention)
        print('MyAttentionWrapper_next_state', next_state)
        
        if self._output_attention:
            if SkipOutput:
                attention = SkipLSTMOutputTuple(attention, state_gate)
            return attention, next_state
        else:
            if SkipOutput:
                cell_output = SkipLSTMOutputTuple(cell_output, state_gate)
            return cell_output, next_state


class SkipDropoutWrapper(DropoutWrapper):
    """Operator adding dropout to inputs and outputs of the given cell."""
    
    @property
    def wrapped_cell(self):
        return self._cell
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)
    
    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        
        print('SkipDropoutWrapper_cell', self._cell)
        print('SkipDropoutWrapper_inputs', inputs)
        print('SkipDropoutWrapper_state', state)
        
        if isinstance(state, list):
            state = tuple(state)
        
        def _should_dropout(p):
            return (not isinstance(p, float)) or p < 1
        
        if _should_dropout(self._input_keep_prob):
            #TODO is only needed if multiple SkipCells are used
            #rebuild = False
            #if isinstance(inputs, SkipLSTMOutputTuple):
            #    inputs, state_gate = inputs
            #    rebuild = True
            print('SkipDropoutWrapper_inputs_sot', inputs)
            inputs = self._dropout(inputs, "input",
                                   self._recurrent_input_noise,
                                   self._input_keep_prob)
        output, new_state = self._cell(inputs, state, scope=scope)
        #if rebuild:
        #    output, new_state_gate = output
        #    output = SkipLSTMOutputTuple(output, new_state_gate)
        
        # Separating SkipState and using the LSTMStateTuple as new_state for Dropout
        if isinstance(self._cell, MultiSkipLSTMCell):
            _, _, up, cup = new_state[-1]
            new_state = [LSTMStateTuple(s.c, s.h) for s in state]
        elif isinstance(self._cell, SkipLSTMCell):
            c, h, up, cup = new_state
            new_state = LSTMStateTuple(c, h)
        print('SkipDropoutWrapper_output', output)
        print('SkipDropoutWrapper_new_state', new_state)
        
        if isinstance(self._cell, SkipLSTMCell):
            output, state_gate = output
        
        if _should_dropout(self._state_keep_prob):
            # Identify which subsets of the state to perform dropout on and
            # which ones to keep.
            shallow_filtered_substructure = nest.get_traverse_shallow_structure(
                self._dropout_state_filter, new_state)
            new_state = self._dropout(new_state, "state",
                                      self._recurrent_state_noise,
                                      self._state_keep_prob,
                                      shallow_filtered_substructure)
            print('SkipDropoutWrapper_new_state', new_state)
        if _should_dropout(self._output_keep_prob):
            output = self._dropout(output, "output",
                                   self._recurrent_output_noise,
                                   self._output_keep_prob)
            print('SkipDropoutWrapper_new_output', output)
        if isinstance(self._cell, MultiSkipLSTMCell):
            final_state = SkipLSTMStateTuple(new_state[-1].c, new_state[-1].h, up, cup)
            new_state[-1] = final_state
            output = SkipLSTMOutputTuple(output, state_gate)
        elif isinstance(self._cell, SkipLSTMCell):
            new_state = SkipLSTMStateTuple(new_state.c, new_state.h, up, cup)
            output = SkipLSTMOutputTuple(output, state_gate)
        print('SkipDropoutWrapper_new_output', output)
        print('SkipDropoutWrapper_new_state', new_state)
        return output, new_state


class SkipMultiRNNCell(MultiRNNCell):
    """RNN cell composed sequentially of multiple simple cells.
    Example:
    ```python
    num_units = [128, 64]
    cells = [BasicLSTMCell(num_units=n) for n in num_units]
    stacked_rnn_cell = MultiRNNCell(cells)
    ```
    """
    
    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum(cell.state_size for cell in self._cells)
    
    @property
    def output_size(self):
        return self._cells[-1].output_size
    
    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._state_is_tuple:
                return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
            else:
                # We know here that state_size of each cell is not a tuple and
                # presumably does not contain TensorArrays or anything else fancy
                return super(MultiRNNCell, self).zero_state(batch_size, dtype)
    
    def call(self, inputs, state):
        print('SkipMultiRNNCell_inputs', inputs)
        print('SkipMultiRNNCell_state', state)
        """Run this multi-layer cell on inputs, starting from state."""
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        for i, cell in enumerate(self._cells):
            print('SkipMultiRNNCell_cell', cell)
            with variable_scope.variable_scope("cell_%d" % i):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, but received: %s" %
                            (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos],
                                                [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                print('SkipMultiRNNCell_cur_inp', cur_inp)
                print('SkipMultiRNNCell_cur_state', cur_state)
                cur_inp, new_state = cell(cur_inp, cur_state)
                print('SkipMultiRNNCell_cur_inp_output', cur_inp)
                print('SkipMultiRNNCell_new_state_output', new_state)
                new_states.append(new_state)
        
        new_states = (tuple(new_states) if self._state_is_tuple else
                      array_ops.concat(new_states, 1))
        
        print('SkipMultiRNNCell_cur_inp', cur_inp)
        print('SkipMultiRNNCell_new_states', new_states)
        return cur_inp, new_states

