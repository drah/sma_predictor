import tensorflow as tf
import helpers

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
        # here
        'global_step': None}

  def train(self, x, y):
    self.sess.run([self._node['global_step'], self._node['train'])

  def predict(self, x):
    raise NotImplementedError

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
      self.sess