import tensorflow as tf
import numpy as np
from tqdm import tqdm


class AutoEncoder:

    def __init__(self, input_shape, n_image_summary=4, use_placeholder=True, name='AutoEncoder', device='/gpu:0'):

        self.device = device
        self.name = name
        self.input_shape = input_shape
        self.use_placeholder = use_placeholder
        self.n_image_summary = n_image_summary

        with tf.device(device), tf.variable_scope(name):
            self._get_input()
            self._build_net()
            self._get_loss()
            self._get_summaries()


    def _get_input(self):
        with tf.variable_scope('Input'):
            if self.use_placeholder:
                self._input = tf.placeholder(tf.float32, shape=[None, *self.input_shape], name='Inputs')
            else:
                raise NotImplementedError

    def _get_summaries(self):
        with tf.variable_scope('Summaries'), tf.device('/cpu:0'):
            tf.summary.scalar('Loss', self._loss)

            tf.summary.image('Input', self.inpt, max_outputs=self.n_image_summary)
            tf.summary.image('reconstructed', self.reconstructed, max_outputs=self.n_image_summary)

            self._summaries = tf.summary.merge_all()


    def _get_loss(self):
        with tf.variable_scope('Loss'):
            self._loss = tf.reduce_mean(tf.squared_difference(self._input, self.reconstructed, name='PixelwiseLoss'), name='MeanLoss')


    def _build_net(self):

        conv_kernels = [
            32, 64, 64,
        ]
        conv_sizes = [
            3, 3, 3
        ]
        conv_paddings = [
            'same', 'same', 'same'
        ]
        fc_sizes = [
            3,
        ]

        with tf.variable_scope('Encoder'):
            hid = tf.layers.conv2d(self._input, conv_kernels[0], conv_sizes[0],padding=conv_paddings[0],
                                   activation=tf.nn.elu, name='conv0')
            for c in range(1, len(conv_kernels)):
                hid = tf.layers.conv2d(hid, conv_kernels[c], conv_sizes[c], padding=conv_paddings[c],
                                       activation=tf.nn.elu, name='conv{}'.format(c))

            downsampled_shape = hid.shape.as_list()

            hid = tf.layers.flatten(hid)
            for fc in range(len(fc_sizes)):
                hid = tf.layers.dense(hid, fc_sizes[fc], activation=tf.nn.elu, name='fc{}'.format(fc))

        with tf.variable_scope('LowDimState'):
            self._low_dim = tf.identity(hid)


        with tf.variable_scope('Decoder'):
            hid = tf.layers.dense(self._low_dim, np.prod(downsampled_shape[1:]))

            hid = tf.reshape(hid, shape=[-1, *downsampled_shape[1:]], name='Deflatten')

            hid = tf.layers.conv2d_transpose(hid, 64, 3, padding='same', activation=tf.nn.relu)
            hid = tf.layers.conv2d_transpose(hid, 64, 3, padding='same', activation=tf.nn.relu)
            hid = tf.layers.conv2d_transpose(hid, self.input_shape[-1], 3, padding='same', activation=tf.nn.relu)


            assert hid.shape[1:] == self.input_shape

        with tf.variable_scope('Reconstructed'):
            self._reconstructed = tf.identity(hid)


    @property
    def inpt(self):
        return self._input

    @property
    def low_dim(self):
        return self._low_dim

    @property
    def reconstructed(self):
        return self._reconstructed

    @property
    def loss(self):
        return self._loss

    @property
    def summaries(self):
        return self._summaries
