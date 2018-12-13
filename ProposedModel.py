import functools
import tensorflow as tf


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceClassification:

    def __init__(self, data, target, maxLength, dropout, useAttention, useBiRNN, num_hidden=200, num_layers=2):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.maxLength = maxLength
        self.dropout = dropout
        self.useAttention = useAttention
        self.useBiRNN = useBiRNN
        self.prediction
        self.error
        self.optimize
        self.attention

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):

        weight, bias = self._weight_and_bias(
            self._num_hidden * 2, int(self.target.get_shape()[1]))
        
        if(self.useAttention):
            prediction = tf.nn.softmax(tf.matmul(self.attention[0], weight) + bias)
        else:
            last = self._last_relevant(self.getOutput, self.length, self)
            prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

        
        return prediction
    
    @lazy_property
    def getOutput(self):
        if(self.useBiRNN):
            
            cellsForward = []
            cellsBackward = []
            
            for i in range(self._num_layers):
                cell = tf.contrib.rnn.GRUCell(self._num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
                cellsForward.append(cell)
            for i in range(self._num_layers):
                cell = tf.contrib.rnn.GRUCell(self._num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
                cellsBackward.append(cell)

            output, *_  = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cellsForward, cellsBackward,
                            inputs=self.data, sequence_length=self.length, dtype=tf.float32)
        else:
            cells = []
        
            for i in range(self._num_layers):
                cell = tf.contrib.rnn.GRUCell(self._num_hidden)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                cells.append(cell)
            cells = tf.contrib.rnn.MultiRNNCell(cells)

            output, _ = tf.nn.dynamic_rnn(
                cells,
                self.data,
                dtype=tf.float32,
                sequence_length=self.length,
            )

        return output
    
    @lazy_property
    def attention(self):
        
        inputs = self.getOutput
        
        if(self.useBiRNN):
            inputs = tf.concat(inputs, 2)
            
        attention_size = 100
        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
           
        return output, alphas
        
        
    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length, self):
        batch_size = tf.shape(output)[0]
        max_length = self.maxLength
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant