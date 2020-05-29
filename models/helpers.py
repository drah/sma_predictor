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

def optimize(loss, learning_rate=2e-4, decay_steps=10000, decay_rate=0.9, var_list=None, name=None):
  with tf.variable_scope(name or 'optimizer'):
    global_step = get_global_step()
    dlr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, True)
    opt = tf.train.RMSPropOptimizer(dlr)
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

def batch_norm(node, is_train, decay_rate=0.96, name='batch_norm'):
  # not tested
  depth = node.shape[-1]
  with tf.variable_scope(name):
    moving_mean = tf.get_variable('moving_mean', [depth], initializer=tf.initializers.zeros(), trainable=False)
    moving_var = tf.get_variable('moving_var', [depth], initializer=tf.initializers.ones(), trainable=False)
    batch_mean, batch_var = tf.nn.moments(node, [i for i in range(len(node.shape)-1)])
    scale = tf.constant(1.)
    offset = tf.constant(0.)
    eps = tf.constant(1e-3)

    inc_rate = 1. - decay_rate
    update_moving_mean = tf.assign(moving_mean, moving_mean * decay_rate + batch_mean * inc_rate)
    update_moving_var = tf.assign(moving_var, moving_var * decay_rate + batch_var * inc_rate)

    def is_train_true():
      with tf.control_dependencies([update_moving_mean, update_moving_var]):
        return tf.nn.batch_normalization(node, batch_mean, batch_var, offset, scale, eps)

    def is_train_false():
      return tf.nn.batch_normalization(node, moving_mean, moving_var, offset, scale, eps)
    
    node = tf.cond(is_train, is_train_true, is_train_false)
  return node