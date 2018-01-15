import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
from tensorflow.python.util import nest

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

try:
  from tensorflow.contrib.layers.python.layers import utils
except:
  from tensorflow.contrib.layers import utils

smart_cond = utils.smart_cond

try:
  LSTMCell = rnn.LSTMCell
  MultiRNNCell = rnn.MultiRNNCell
  dynamic_rnn_decoder = seq2seq.dynamic_rnn_decoder
  simple_decoder_fn_train = seq2seq.simple_decoder_fn_train
except:
  LSTMCell = tf.contrib.rnn.LSTMCell
  MultiRNNCell = tf.contrib.rnn.MultiRNNCell
  dynamic_rnn_decoder = tf.contrib.seq2seq.dynamic_rnn_decoder
  simple_decoder_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train

try:
  from tensorflow.python.ops.gen_array_ops import _concat_v2 as concat_v2
except:
  concat_v2 = tf.concat_v2

def decoder_rnn(cell, inputs,
                enc_outputs, enc_final_states,
                seq_length, hidden_dim,
                num_glimpse, batch_size, is_train,
                end_of_sequence_id=0, initializer=None,
                max_length=None, dec_targets = None):
  with tf.variable_scope("decoder_rnn") as scope:
    def attention(ref, query, with_softmax, scope="attention"):
      with tf.variable_scope(scope):
        W_ref = tf.get_variable(
            "W_ref", [1, hidden_dim, hidden_dim], initializer=initializer)
        W_q = tf.get_variable(
            "W_q", [hidden_dim, hidden_dim], initializer=initializer)
        v = tf.get_variable(
            "v", [hidden_dim], initializer=initializer)

        encoded_ref = tf.nn.conv1d(ref, W_ref, 1, "VALID", name="encoded_ref")
        encoded_query = tf.expand_dims(tf.matmul(query, W_q, name="encoded_query"), 1)
        tiled_encoded_Query = tf.tile(
            encoded_query, [1, tf.shape(encoded_ref)[1], 1], name="tiled_encoded_query")
        scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1])
        
        #paddings = tf.constant([[0,0],[0,hidden_dim-max_length]])
        #padding_scores = tf.pad(scores, paddings, 'CONSTANT')
        #tf.reshape(padding_scores, shape=[batch_size, hidden_dim])

        if with_softmax:
          return tf.nn.softmax(scores)
        else:
          return scores

    def glimpse(ref, query, scope="glimpse"):
      p = attention(ref, query, with_softmax=True, scope=scope)
      alignments = tf.expand_dims(p, 2)
      return tf.reduce_sum(alignments * ref, [1])

    def output_fn(ref, query, mask, num_glimpse):
      if query is None:
        #print "query is none" 
        return tf.zeros([max_length], tf.float32) # only used for shape inference
      else:
        for idx in range(num_glimpse):
          query = glimpse(ref, query, "glimpse_{}".format(idx))
        if mask is not None:
          return attention(ref, query, with_softmax=False, scope="attention") - 1000000 * mask
          #return attention(ref, query, with_softmax=False, scope="attention")
        else:
          return attention(ref, query, with_softmax=False, scope="attention")


    def input_fn(sampled_idx):
      return tf.stop_gradient(
          tf.gather_nd(enc_outputs, index_matrix_to_pairs(sampled_idx)))

    if is_train:
      decoder_fn = simple_decoder_fn_train(enc_final_states)
    else:
      maximum_length = tf.convert_to_tensor(max_length, tf.int32)

      def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        cell_output = output_fn(ref=enc_outputs, query=cell_output, num_glimpse=num_glimpse, mask=None)
        if cell_state is None:
          cell_state = enc_final_states
          next_input = cell_input
          done = tf.zeros([batch_size,], dtype=tf.bool)
        else:
          sampled_idx = tf.cast(tf.argmax(cell_output, 1), tf.int32)
          next_input = input_fn(sampled_idx)
          done = tf.equal(sampled_idx, end_of_sequence_id)

        done = tf.cond(tf.greater(time, maximum_length),
          lambda: tf.ones([batch_size,], dtype=tf.bool),
          lambda: done)
        return (done, cell_state, next_input, cell_output, context_state)

    print "before dynamic_rnn_decode"
    outputs, final_state, final_context_state = \
        dynamic_rnn_decoder(cell, decoder_fn, inputs=inputs,
                            sequence_length=seq_length, scope=scope)
    print "after dynamic_rnn_decode"

    if is_train:
      transposed_outputs = tf.transpose(outputs, [1, 0, 2])   # seq_length * batch_size * hidden
      transposed_masks = mask(dec_targets, max_length)    # seq_length * batch-size * max_length
      fn = lambda x: output_fn(enc_outputs, x[0], x[1], num_glimpse)
      outputs = tf.transpose(tf.map_fn(fn, (transposed_outputs, transposed_masks), dtype=tf.float32), [1, 0, 2])

    return outputs, final_state, final_context_state     # outputs: seq_length * batch_size * seq_length

def mask(dec_targets, seq_length):
    # acording to dec_targets, gen mask,  which nodes has visited and will not to be visited again
    with tf.Session() as sess:
        onehot_dec_targets = tf.one_hot(dec_targets, seq_length)  # batch_size * seq_length * seq_length
        mask = tf.cumsum(onehot_dec_targets, axis=1, exclusive=True) # cumulate add front subtensor, e.g. b[i] = b[i] + b[i-1]
        shape = tf.shape(dec_targets)
        mask = tf.zeros([shape[0], shape[1], seq_length], tf.float32)
        # whose shape is equal with query
        transposed_mask = tf.transpose(mask, [1,0,2])   #  seq_length * batch_size * seq_length
        print sess.run(tf.shape(transposed_mask))
    return transposed_mask

def trainable_initial_state(batch_size, state_size,
                            initializer=None, name="initial_state"):
  flat_state_size = nest.flatten(state_size)

  if not initializer:
    flat_initializer = tuple(tf.zeros_initializer() for _ in flat_state_size)
  else:
    flat_initializer = tuple(tf.zeros_initializer() for initializer in flat_state_size)

  names = ["{}_{}".format(name, i) for i in xrange(len(flat_state_size))]
  tiled_states = []

  for name, size, init in zip(names, flat_state_size, flat_initializer):
    shape_with_batch_dim = [1, size]
    initial_state_variable = tf.get_variable(
        #name, shape=shape_with_batch_dim, initializer=init())
        name, initializer=init(shape_with_batch_dim))

    tiled_state = tf.tile(initial_state_variable,
                          [batch_size, 1], name=(name + "_tiled"))
    tiled_states.append(tiled_state)

  return nest.pack_sequence_as(structure=state_size,
                               flat_sequence=tiled_states)

def index_matrix_to_pairs(index_matrix):
  # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
  #                        [[0, 2], [1, 3], [2, 1]]]
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
  rank = len(index_matrix.get_shape())
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, axis=1),
        [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=rank)
