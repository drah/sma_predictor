import tensorflow as tf
from . import helpers
from .base import Base

class NNRegression(Base):
  def __init__(self, input_dim, ckpt=None, save_dir='log', name='NNRegression'):
    super(NNRegression, self).__init__()
    self._in_dim = input_dim
    self._ckpt = ckpt
    self._save_dir = save_dir
    self._name = name

    self._build()

  def _build(self):
    with tf.variable_scope(self._name):
      input = tf.placeholder(tf.float32, shape=[None, self._in_dim])
      net = input

      net = helpers.dense(net, 32, name='dense1')
      net = tf.nn.relu(net)
      net = helpers.dense(net, 1, name='output')
      output = net

    with tf.variable_scope('training'):
      label = tf.placeholder(tf.float32, shape=[None])
      l2_loss = helpers.l2_loss(tf.expand_dims(label, 1), output)
      global_step, train_step = helpers.optimize(l2_loss)

    with tf.name_scope('metric'):
      metric = helpers.metric(label, output)

    self._node['in'] = input
    self._node['out'] = output
    self._node['label'] = label
    self._node['global_step'] = global_step
    self._node['train'] = train_step
    self._node['loss'] = l2_loss
    self._node['metric'] = metric

if __name__ == '__main__':
  nn_reg = NNRegression(5)
