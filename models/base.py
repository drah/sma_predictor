from pprint import pprint
import tensorflow as tf
from . import helpers

class Base:
  def __init__(self):
    self._sess = None
    self._saver = None
    self._save_dir = None
    self._ckpt = None
    self._node = {
        'in': None,
        'out': None,
        'train': None,
        'loss': None,
        'metric': None,
        'global_step': None}
    self._train_run = {
      'global_step': self._node['global_step'],
      'train': self._node['train'],
      'loss': self._node['loss'],
      'metric': self._node['metric']}
    self._validate_run = {
      'loss': self._node['loss'],
      'metric': self._node['metric']}

  def fit(
      self, train, val, start_step=0, end_step=100000,
      batch_size=100, val_step=100, val_batch_size=100, es_tolerance=10):
    best_metric = float('-inf')
    miss = 0
    for step in range(start_step, end_step):
      batch_x, batch_y = train.get(batch_size)
      train_log = self.train(batch_x, batch_y)
      if step % val_step == 0:
        val_batch_x, val_batch_y = val.get(val_batch_size)
        val_log = self.validate(val_batch_x, val_batch_y)
        pprint(train_log)
        pprint(val_log)
        print()
        if val_log['metric'] > best_metric:
          best_metric = val_log['metric']
          miss = 0
        else:
          miss += 1
          if miss == es_tolerance:
            print("Early stopping at step %d" % step)
            break

  def evaluation(self, data, batch_size=100):
    pass

  def train(self, x, y):
    return self.sess.run(
        self._train_run,
        {self._node['in']: x, self._node['out']: y})

  def validate(self, x, y):
    return self.sess.run(
        self._validate_run,
        {self._node['in']: x, self._node['out']: y})

  def predict(self, x):
    return self.sess.run(
        self._node['out'],
        {self._node['in']: x})

  def _build(self):
    raise NotImplementedError

  @property
  def sess(self):
    if self._sess is None:
      self._sess = tf.Session()
    return self._sess

  @property
  def saver(self):
    helpers.handle_dir(self._save_dir)
    if self._saver is None:
      self._saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
    return self._saver

  def save(self):
    self.saver.save(self.sess, self._save_dir + 'ckpt')

  def _init_session(self):
    if self._ckpt is not None:
      self.saver.restore(self.sess, self._ckpt)
    else:
      self.sess.run(tf.global_variables_initializer())