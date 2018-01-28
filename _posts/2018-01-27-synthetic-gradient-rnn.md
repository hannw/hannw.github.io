---
layout: post
title: "How to Implement Synthetic Gradient for RNN in Tensorflow"
description: "The theory of synthetic gradient for RNN, it's implications, and how to implement it in tensorflow"
tags: [metalearning, machinelearning]
---

Synthetic gradient, or the decoupled neural interface (DNI), was probably the most exciting paper I read about since last year. Synthetic gradient manages to decouple all layers of a deep network, making asynchronous training possible. Furthermore, it unifies the theory of reinforcement learning and supervised training into a single framework.

In terms of recurrent neural network, it has some other important implications. Before, while training a RNN with long time span, we can only use truncated back propagation through time. It has been shown that being able to persist the hidden state through time helps with the trianing. However, without propagating the gradient from the future, the error signal cannot reach and correct the mistake made a long time ago. Now, by using synthetic gradient, we can create an unbiased estimate of the gradient, therefore making the back propagation through time of a infinite long RNN possible.


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

For more information about the state saving queue, check out the RNN tutorial in [Tensorflow Dev Summet 2017](https://www.youtube.com/watch?v=RIR_-Xlbp7s).

Note that in the original paper, they store \\(\Delta t + 1\\) time span for each \\(\Delta t\\) time span. I chose to store \\(2 \Delta t\\) inputs and labels, since the future information \\(\Delta t \leq t < 2\Delta t\\) could be useful for computing synthetic gradient, if you have other customized architecture of DNI.

## Stitching The Networks Together

The hardest part in implementing the synthetic gradient is to figure out which exactly hidden states to use while calculating the synthetic gradient and it's target. To make the steps slightly clearer for the sack of implementation, let's rederive the equations. The total gradient of a RNN can be written as the follwing.

{% raw %}
$$
\begin{align*}
\sum^{\infty}_{\tau=t} \frac{\partial L_\tau}{\partial \theta}
&= \sum^{t + \Delta t}_{\tau = t} \frac{\partial L_\tau}{\partial \theta} + (\sum^{\infty}_{\tau = t + \Delta t} \frac{\partial L_\tau}{\partial h_{t + \Delta t -1}}) \frac{\partial h_{t+ \Delta t - 1}}{\partial \theta} \\
&=\sum^{t + \Delta t}_{\tau = t} \frac{\partial L_\tau}{\partial \theta} + \delta_{t + \Delta t} \frac{\partial h_{t + \Delta t - 1}}{\partial \theta}
\end{align*}$$
{% endraw %}

where the target gradient is defined as
{% raw %}
$$
\delta_t \equiv \sum^{\infty}_{\tau = t} \frac{\partial L_\tau}{\partial h_{t -1}}
$$
{% endraw %}

We can ask the rnn to synthesize the gradient using a linear approximator.
{% raw %}
$$
\hat{\delta}_t = f(h_t) = W h_t + b
$$
{% endraw %}

Since the target gradient is not tracktable, we can bootstrap the gradient using the following trick. Suppose in a time span, \\(\Delta t\\),

{% raw %}
$$
\delta_t = \sum_{\tau=t}^{t+\Delta t -1} \frac{\partial L_\tau}{\partial h_{t-1}} + \delta_{t + \Delta t}\frac{\partial h_{t + \Delta t - 1}}{\partial h_{t-1}}
$$
{% endraw %}

where \\(h_{t-1}\\) is the initial hidden state, and \\(h_{t + \Delta t-1}\\) is the final state in the time span. This formula says that the target for synthetic gradient is the gradient of the loss in the time span with respect to the initial state, plus the synthetic gradient in the next time span times the derivative of the last hidden state with respect to the initial state.