---
layout: post
title: "How to Implement Synthetic Gradient for RNN in Tensorflow"
description: "The theory of synthetic gradient for RNN, it's implications, and how to implement it in tensorflow"
tags: [metalearning, machinelearning]
---

Synthetic gradient, or the decoupled neural interface (DNI), was probably the best paper I read about since last year. Synthetic gradient manages to decouple all layers of a deep network, making asynchronous training possible. Furthermore, it unifies the theory of reinforcement learning and supervised training into a single framework.

In terms of recurrent neural network, it has some other important implication. Before we had synthetic gradient, while training a RNN with long time span, we can only resort to truncated back propagation through time. It has been shown that being able to persist the hidden state through time helps with the trianing. However, without propagating the gradient from the future, the error signal cannot reach an correct the mistake made a long time ago. Now, by using synthetic gradient, we can create an unbiased estimate of the gradient, therefore making the back propagation through time of a infinite long RNN possible.

Given all these theoretical benefit, I cannot wait to jump on to the bandwagon and start training all my RNNs by synthetic gradient, but I just could not find an open source implementation of synthetic gradient written in tensorflow. After some digging, here is one implementation I came up with.

## Preparing the Training Data

Synthetic gradient for RNN is slightly different from that of a feedforward neural network. In a long RNN, the state needs to be saved for the next time step. Also, since calculating the target for synthetic gradient is dependent on future time step, the inputs in the future also needs to be fed during training time. Let's see how we can implement this in practice, using the PTB dataset.

```python
def pdb_state_saver(raw_data, batch_size, num_steps, init_states,
  num_unroll, num_threads=3, capacity=1000, allow_small_batch=False,
                    epoch=1000, name=None):
  data_len = len(raw_data)
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    n_seq = (data_len - 1) // num_steps
    
    # prepare the input data x and output label y as tensors 
    raw_data_x = tf.reshape(raw_data[0 : n_seq * num_steps],
                      [n_seq, num_steps])
    raw_data_y = tf.reshape(raw_data[1 : (n_seq * num_steps + 1)],
                      [n_seq, num_steps])
    
    # we prepare the future intputs and labels by rolling the data
    next_raw_data_x = _circular_shift(raw_data_x, num_unroll, axis=1)
    next_raw_data_y = _circular_shift(raw_data_y, num_unroll, axis=1)

    # the data are fed to tensorflow by the Dataset API
    keys = tf.convert_to_tensor(
      ['seq_{}'.format(i) for i in range(n_seq)], name="key", dtype=tf.string)
    seq_len = tf.tile([num_steps], [n_seq])
    data = tf.data.Dataset.from_tensor_slices(
      (keys, raw_data_x, next_raw_data_x, raw_data_y, next_raw_data_y, seq_len))
    data = data.repeat(epoch)
    iterator = data.make_one_shot_iterator()

    ......
```

After preparing the data, we feed the data points into state saving queue.
```python
    next_key, next_x, next_next_x, next_y, next_next_y, next_len = iterator.get_next()
    seq_dict = {'x':next_x, 'next_x':next_next_x, 'y':next_y, 'next_y':next_next_y}
    # The following will instantiate a `NextQueuedSequenceBatch` as state saver
    batch = batch_sequences_with_states(
      input_key=next_key,
      input_sequences=seq_dict,
      input_context={},
      input_length=next_len,
      initial_states=init_states,
      num_unroll=num_unroll,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=capacity,
      allow_small_batch=allow_small_batch,
      pad=True)
  return batch
```