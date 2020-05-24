import tensorflow as tf 
import os

def dense(node, n, act_fn=None, name=None):
  with tf.variable_scope(name):
    w = get_weight([node.shape[-1], n])
    b = get_bias([n])

  with tf.name_scope(name):
    node = tf.matmul(node, w) + b

  return node

def conv2d(node, n_filter, k_size, strides, padding, name=None):
  with tf.variable_scope(name):
    w = get_weight([k_size[0], k_size[1], node.shape[-1], n_filter])
    b = get_bias([n_filter])

  with tf.name_scope(name):
    node = tf.nn.conv2d(node, w, [1, strides[0], strides[1], 1], padding)

  return node

def optimize(loss, learning_rate=1e-3, decay_steps=10000, decay_rate=0.96, var_list=None, name=None):
  with tf.variable_scope(name or 'optimizer'):
    global_step = get_global_step()
    dlr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, True)
    opt = tf.train.AdamOptimizer(dlr)
    train_step = opt.minimize(loss, global_step, var_list)
  return global_step, train_step

def get_global_step():
  return tf.get_variable('global_step', shape=[], initializer=tf.initializers.constant(0, tf.int64), trainable=False)

def get_weight(shape, name=None):
  w = tf.get_variable(name or 'weight', shape, initializer=tf.initializers.he_uniform())
  return w

def get_bias(shape, init_value=0.2, name=None):
  b = tf.get_variable(name or 'bias', shape, initializer=tf.initializers.constant(init_value))
  return b

def get_saver(var_list=None, max_to_keep=10):
  return tf.train.Saver(var_list, max_to_keep=max_to_keep)

def handle_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

def l2_loss(label, pred):
  return tf.reduce_mean(tf.squared_difference(label, pred))

def metric(label, pred):
  return tf.sqrt(l2_loss(label, pred))