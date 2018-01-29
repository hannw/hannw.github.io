---
layout: post
title: "How to Implement Synthetic Gradient for RNN in Tensorflow"
description: "The theory of synthetic gradient for RNN, it's implications, and how to implement it in tensorflow"
tags: [metalearning, machinelearning]
---

Here I described a way to implement [Synthetic Gradient](https://arxiv.org/abs/1608.05343) for RNN in tensorflow, and the intuition behind it. For the full implementation, please check out my [github repo](https://github.com/hannw/sgrnn).

Synthetic gradient, or the decoupled neural interface (DNI), was probably the most exciting paper I read about last year. Synthetic gradient manages to decouple all layers of a deep network, making asynchronous training possible. Furthermore, it unifies the theory of reinforcement learning and supervised training into a single framework.

| ![sgrnn.gif](https://storage.googleapis.com/deepmind-live-cms-alt/documents/3-10_18kmHY7.gif) | 
|:--:| 
| *Sythetic Gradient RNN in action. From Deepmind [blog post](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/). Courtesy of Jaderberg et al.* |

When applied to recurrent neural network, it has some other important implications. I often have to deal with recurrent neural network that runs thousands of time steps at work. To train a RNN with such a long time horizon, we can only use truncated back propagation through time (TBPTT), due to memory constraint of caching all time steps. It has been shown that being able to persist the hidden state through time helps with the trianing. However, without propagating the gradient from the future, the error signal cannot reach and correct the mistake made a long time ago, so a long range dynamics cannot truely be learnt. Now, by using synthetic gradient, we can create an unbiased estimate of the true gradient, therefore making back propagation through time to infinity possible.

Given all the theoretical benefit, I cannot wait to start training all my RNNs by synthetic gradient, but most of the synthetic gradient online only covers FCN. So, I decided to write my own version for RNN.

## Preparing the Training Data

Synthetic gradient for RNN is slightly different from that of a feedforward neural network. In a RNN, the hidden states needs to be saved for the next time step. Also, since calculating the target for synthetic gradient is dependent on one future time step, the inputs in the future also needs to be fed during training time. Let's see how we can implement this in practice, using the PTB dataset.

{% highlight python %}
def pdb_state_saver(raw_data, batch_size, num_steps, init_states,
                    num_unroll, num_threads=3, capacity=1000,
                    allow_small_batch=False, epoch=1000, name=None):
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
{% endhighlight %}

After preparing the data, we feed the data points into state saving queue. This will produce a `NextQueuedSequenceBatch` object which contains the state saver, and the properly batched sequences.

{% highlight python %}
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
{% endhighlight %}

For more information about the state saving queue, check out the RNN tutorial in [Tensorflow Dev Summet 2017](https://www.youtube.com/watch?v=RIR_-Xlbp7s).

Note that in the original paper, they store \\(\Delta t + 1\\) time span for each \\(\Delta t\\) time span. I chose to store \\(2 \Delta t\\) inputs and labels, since the future information \\(\Delta t \leq t < 2\Delta t\\) could be useful for computing synthetic gradient, if you have other customized architecture of DNI.

## Stitching The Networks Together

Perhaps the hardest part in implementing the synthetic gradient is to figure out which exactly hidden states to use while calculating the synthetic gradient. To make the steps slightly clearer for the sack of implementation, let's rederive the equations. The total gradient of a RNN is

{% raw %}
$$
\begin{align}
\sum^{\infty}_{\tau=t} \frac{\partial L_\tau}{\partial \theta}
&= \sum^{t + \Delta t}_{\tau = t} \frac{\partial L_\tau}{\partial \theta} + (\sum^{\infty}_{\tau = t + \Delta t} \frac{\partial L_\tau}{\partial h_{t + \Delta t -1}}) \frac{\partial h_{t+ \Delta t - 1}}{\partial \theta} \\
&=\sum^{t + \Delta t}_{\tau = t} \frac{\partial L_\tau}{\partial \theta} + \delta_{t + \Delta t} \frac{\partial h_{t + \Delta t - 1}}{\partial \theta} \tag{1}\label{eq:sgderive}
\end{align}
$$
{% endraw %}

, where the future gradient is defined as
{% raw %}
$$
\delta_t \equiv \sum^{\infty}_{\tau = t} \frac{\partial L_\tau}{\partial h_{t -1}}.
$$
{% endraw %}
Note that the total gradient of an RNN is just the local gradient in time span \\(\Delta t\\), plus the future gradient term. Instead of computing the future gradient, we can ask the rnn to synthesize the gradient using a linear approximator.
{% raw %}
$$
\hat{\delta}_t = f(h_t) = W h_t + b \tag{2}\label{eq:sg}
$$
{% endraw %}

Now, it should become clear that the entire point of synthetic gradient, is to make the network guess the future gradient for you, so you can train the network without knowing any future data.

It all sounds too good to be true, except that we can actually make the network do educated guess by supervised training. If we need to compute the infinitely long future gradient, aren't we back to square one? Since the target gradient is not tracktable, we can use a little trick to bootstrap the gradient. Suppose in a time span, \\(\Delta t\\),

{% raw %}
$$
\delta_t = \sum_{\tau=t}^{t+\Delta t -1} \frac{\partial L_\tau}{\partial h_{t-1}} + \delta_{t + \Delta t}\frac{\partial h_{t + \Delta t - 1}}{\partial h_{t-1}} \tag{3}\label{eq:targetgrad}
$$
{% endraw %}

where \\(h_{t-1}\\) is the initial hidden state, and \\(h_{t + \Delta t-1}\\) is the final state in the time span. This formula says that the target for synthetic gradient is the gradient of the loss in the time span with respect to the initial state, plus the synthetic gradient in the next time span times the derivative of the last hidden state with respect to the initial state. This bootstrap procedure is the part that unifies supervised training with reinforcement learning. The bootstrap procedure is analogous to TD(\\(\lambda\\)).

### Propagate RNN State

The synthetic graident is produced by the first core of RNN at each time chunk, \\(\Delta t\\). We use a simple dense layer, or an `OutputProjectionWrapper`, to compute the synthetic gradient. The dense layer produce both the output logits and the synthetic graident.
{% highlight python %}
    self._cell = tf.contrib.rnn.OutputProjectionWrapper(
        self.base_cell, self.output_size + self.total_state_size)
{% endhighlight %}

Next we use the `static_state_saving_rnn` to propagate the RNN core through \\(\Delta t\\), and save the final state of the RNN to the state saver.
{% highlight python %}
def build_synthetic_gradient_rnn(self, inputs, sequence_length):
  with tf.name_scope('RNN'):
    inputs = tf.unstack(inputs, num=self.num_unroll, axis=1)
    outputs, final_state = tf.nn.static_state_saving_rnn(
      cell=self.cell,
      inputs=inputs,
      state_saver=self.state_saver,
      state_name=self.state_name,
      sequence_length=sequence_length)

    with tf.name_scope('synthetic_gradient'):
      synthetic_gradient = tf.slice(
        outputs[0], begin=[0, self.output_size], size=[-1, -1])
      synthetic_gradient = tf.split(
        synthetic_gradient, nest.flatten(self.state_size), axis=1)

    with tf.name_scope('logits'):
      stacked_outputs = tf.stack(outputs, axis=1)
      logits = tf.slice(stacked_outputs, begin=[0, 0, 0], size=[-1, -1, self.output_size])

  return logits, final_state, synthetic_gradient
{% endhighlight %}

### Bootstrap the Target of Synthetic Gradient

The synthetic gradient in the next \\(\Delta t\\) time span needs to be computed in order for us to bootstrap the target synthetic gradient. Fortunately, it only depends on the first core of the next time span.
{% highlight python %}
  def build_next_synthetic_gradient(self, final_state, next_inputs):
    with tf.name_scope('next_synthetic_gradient'):
      next_inputs = tf.unstack(next_inputs, num=self.num_unroll, axis=1)
      next_output, _ = self.cell(next_inputs[0], final_state)
      next_synthetic_gradient = tf.slice(
        next_output, begin=[0, self.output_size], size=[-1, -1])
      next_synthetic_gradient = tf.split(
        next_synthetic_gradient, nest.flatten(self.state_size), axis=1)
    return next_synthetic_gradient
{% endhighlight %}

Next, we use the next synthetic gradient to bootstrap the target synthetic gradient like in equation \eqref{eq:targetgrad}.
{% highlight python %}
  def sg_target(self, loss, next_sg, final_state):
    local_grad = tf.gradients(ys=loss, xs=nest.flatten(self.init_state))
    next_sg = [tf.where(self.is_done, tf.zeros_like(grad), grad) for grad in next_sg]
    future_grad = tf.gradients(
      ys=nest.flatten(final_state),
      xs=nest.flatten(self.init_state),
      grad_ys=next_sg)
    # for two sequence, the target is bootstrapped
    # at the end sequence, the target is only single sequence
    sg_target = [tf.stop_gradient(tf.add(lg, fg))
      for lg, fg in zip(local_grad, future_grad)]
    return sg_target
{% endhighlight %}

### Compute the gradient for the RNN
To compute the gradient in current \\(\Delta t\\) to train RNN like in equation \eqref{eq:sgderive}, we compute the total gradient by adding the local gradient with the future graidients, which is just the synthetic gradient multiplied by the gradient of the final state. We can pass it in to `tf.gradients` using the `grad_ys` argument. 
{% highlight python %}
  def gradient(self, loss, tvars, next_sg, final_state):
    grad_local = tf.gradients(ys=loss, xs=tvars, grad_ys=None,
                              name='local_gradients')
    received_sg = [tf.where(self.is_done, tf.zeros_like(nsg), nsg) for nsg in next_sg]
    grad_sg = tf.gradients(
      ys=nest.flatten(final_state), xs=tvars, grad_ys=received_sg,
      name='synthetic_gradients')
    grad = [tf.add(gl, gs) if gs is not None else gl for gl, gs in zip(grad_local, grad_sg)]
    return grad
{% endhighlight %}

### Compute the gradient for Synthetic Gradient Core
To train the RNN for synthetic gradient, we just compute the mean squared loss between the bootstrapped target and the predicted synthetic gradient.
{% highlight python %}
    sg_target = self.sg_target(loss, next_sg, final_state)
    sg_loss = tf.losses.mean_squared_error(labels=tf.stack(sg_target), predictions=tf.stack(sg))
    sg_grad = tf.gradients(ys=sg_loss, xs=tvars)
{% endhighlight %}

Once you have the gradient of the loss signal and the synthetic gradient, you can start training your network. Just pick your favorite optimizer, and enjoy the benefit of propagating gradient for infitie number of steps. One thing to note is, since RNN operates in time domain, the synthetic gradient, in this case, does not give you the benefit of decoupling and asynchronous training of layers. The network still need to propagate the hidden states step by step. Not that I am complaining though.