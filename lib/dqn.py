import tensorflow as tf


class DQN:
    def __init__(self, input_len, output_len, batch_size):
        self.input_len = input_len
        self.output_len = output_len
        self.batch_size = batch_size

        # basic networke we're learning
        self.input_t = tf.placeholder(tf.float32, shape=(self.batch_size, self.input_len), name="input")

        # second net, we use to make a forecasts.
        self.next_input_t = tf.placeholder(tf.float32, shape=(self.batch_size, self.input_len), name="next_input")

    def __str__(self):
        return """DQN: input={input_len}, output={output_len}, batch={batch_size}
    input_t={input_t}
    next_input_t={next_input_t}
        """.format(input_len=self.input_len, output_len=self.output_len, batch_size=self.batch_size,
                   input_t=self.input_t, next_input_t=self.next_input_t)


class DenseDQN(DQN):
    def __init__(self, input_len, output_len, batch_size, neurons, dropout_keep_prob=0.5):
        DQN.__init__(self, input_len, output_len, batch_size)
        self.neurons = neurons
        self.dropout_keep_prob = dropout_keep_prob

        self.qvals_t = self._make_forward_net(self.input_t, is_trainable=True)
        self.next_qvals_t = self._make_forward_net(self.next_input_t, is_trainable=False)

    def __str__(self):
        return DQN.__str__(self) + "\n" + \
        """DenseDQN: neurons={neurons}, dropout={dropout_keep_prob}
    qvals_t={qvals_t}
    next_qvals_t={next_qvals_t}
        """.format(neurons=self.neurons, dropout_keep_prob=self.dropout_keep_prob,
                   qvals_t=self.qvals_t, next_qvals_t=self.next_qvals_t)

    def _make_forward_net(self, input_t, is_trainable):
        w_attrs = {'trainable': is_trainable, 'name': 'w'}
        b_attrs = {'trainable': is_trainable, 'name': 'b'}

        if is_trainable:
            init = tf.contrib.layers.xavier_initializer()
            suff = "_T"
        else:
            init = tf.zeros
            suff = "_R"

        with tf.name_scope("L0" + suff):
            w = tf.Variable(init((self.input_len, self.neurons[0])), **w_attrs)
            b = tf.Variable(tf.zeros((self.neurons[0], )), **b_attrs)
            v = tf.matmul(input_t, w) + b
            l0_out = tf.nn.relu(v, name="L0")
            if is_trainable:
                tf.contrib.layers.summarize_activation(l0_out)
        layer_output = l0_out

        for layer_index in range(len(self.neurons)-1):
            layer_input = layer_output
            with tf.name_scope("L{}{}".format(layer_index, suff)):
                w = tf.Variable(init((self.neurons[layer_index], self.neurons[layer_index+1])), **w_attrs)
                b = tf.Variable(tf.zeros((self.neurons[layer_index+1], )), **b_attrs)

                v = tf.matmul(layer_input, w) + b
                layer_output = tf.nn.relu(v, name="L{}".format(layer_index))
                if is_trainable:
                    if self.dropout_keep_prob is not None and self.dropout_keep_prob < 1.0:
                        layer_output = tf.nn.dropout(layer_output, self.dropout_keep_prob)
                    tf.contrib.layers.summarize_activation(layer_output)

        with tf.name_scope("LOut" + suff):
            w = tf.Variable(init((self.neurons[-1], self.output_len)), **w_attrs)
            b = tf.Variable(tf.zeros((self.output_len, )), **b_attrs)
            v = tf.matmul(layer_output, w) + b
            layer_output = tf.identity(v, "out")

        return layer_output