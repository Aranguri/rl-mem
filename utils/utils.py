import tensorflow as tf
import gzip
import pickle as cPickle
import os
import sys
import numpy as np
from tqdm import tqdm

def get_data_mnist():
    f = gzip.open('{}/../data/mnist.pkl.gz'.format(os.path.dirname(__file__)), 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')
        f.close()
    (x_train, _), (x_test, _) = data

    return x_train, x_test

def get_optimizer(optim, **optim_args):
    if optim == 'adam':
        return tf.train.AdamOptimizer(**optim_args)
    else:
        raise NotImplementedError

def create_logdir(logdir, run_name):
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    if run_name is None:
        run_name = 'Run{}'.format(len(os.listdir(logdir)))

    path = logdir + '/' + run_name
    os.makedirs(path, exist_ok=True)
    return path


def train_mnist(network, n_iters, logdir='../logs', run_name=None, optimizer='adam', **optim_args):

    x_train, x_test = get_data_mnist()
    examples, _, _  = x_train.shape

    optim = get_optimizer(optimizer, **optim_args)
    train_op = optim.minimize(network.loss)

    logdir = os.path.dirname(__file__) + '/' + logdir
    path = create_logdir(logdir, run_name)

    file_writer = tf.summary.FileWriter(path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        file_writer.add_graph(sess.graph)

        for iteration in tqdm(range(n_iters)):
            batch_inds = np.random.choice(np.arange(examples), 32)
            batch = x_train[batch_inds]
            batch = np.reshape(batch, (32, 28, 28, 1))


            _, summ = sess.run([train_op, network.summaries], feed_dict={
                network.inpt: batch
            })

            if iteration % 100 == 0:
                file_writer.add_summary(summ, global_step=iteration)
