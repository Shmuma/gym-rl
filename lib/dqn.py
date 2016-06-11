import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, input_len, output_len, batch_size, gamma_initial = 0.99):
        self.input_len = input_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.gamma_t = tf.Variable(initial_value=gamma_initial, trainable=False, dtype=tf.float32)

        self.state_t = tf.placeholder(tf.float32, shape=(None, self.input_len), name="input")
        self.actions_t = tf.placeholder(tf.int32, shape=(None, 1), name="actions")
        self.rewards_t = tf.placeholder(tf.float32, shape=(None, 1), name="Rewards")
        self.next_state_t = tf.placeholder(tf.float32, shape=(None, self.input_len), name="next_input")

        self.qvals_t = self._make_network(self.state_t, is_trainable=True)
        self.next_qvals_t = self._make_network(self.next_state_t, is_trainable=False)
        self.loss_t = self._make_loss()

    def __str__(self):
        return """DQN: input={input_len}, output={output_len}, batch={batch_size}
    input_t={input_t}
    qvals_t={qvals_t}
    next_input_t={next_input_t}
    next_qvals_t={next_qvals_t}
        """.format(input_len=self.input_len, output_len=self.output_len, batch_size=self.batch_size,
                   input_t=self.state_t, next_input_t=self.next_state_t,
                   qvals_t=self.qvals_t, next_qvals_t=self.next_qvals_t)

    def calc_qvals(self, state):
        dims = np.ndim(state)
        if dims == 1:
            state = [state]
        qvals, = tf.get_default_session().run([self.next_qvals_t], feed_dict={
            self.next_state_t: state
        })
        if dims == 1:
            return qvals[0]
        return qvals

    def _make_network(self, input_tensor, is_trainable):
        raise NotImplementedError

    def _make_loss(self):
        # make one-hot mask from actions
        with tf.name_scope("loss"):
            mask_t = tf.one_hot(self.actions_t, self.output_len, on_value=1, off_value=0, dtype=tf.int32)
            max_reward_t = tf.reduce_max(self.next_qvals_t, reduction_indices=1, keep_dims=True)
            qbellman_t = self.rewards_t * mask_t + max_reward_t * mask_t * self.gamma_t
            return tf.nn.l2_loss(self.qvals_t * mask_t - qbellman_t)


class DenseDQN(DQN):
    def __init__(self, input_len, output_len, batch_size, neurons, dropout_keep_prob=0.5):
        self.neurons = neurons
        self.dropout_keep_prob = dropout_keep_prob
        DQN.__init__(self, input_len, output_len, batch_size)

    def __str__(self):
        return DQN.__str__(self) + "\n" + \
        """DenseDQN: neurons={neurons}, dropout={dropout_keep_prob}
        """.format(neurons=self.neurons, dropout_keep_prob=self.dropout_keep_prob)

    def _make_network(self, input_t, is_trainable):
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

