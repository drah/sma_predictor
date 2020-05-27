from sklearn.utils import shuffle

class Data:
  def __init__(self, data, labels):
    self._data = data
    self._labels = labels
    self.i = 0

  def get(self, batch_size):
    self.i += batch_size
    if self.i > len(self.data):
      self._data, self._labels = shuffle(self._data, self._labels)
      self.i = batch_size
    return self.data[(self.i - batch_size): self.i], self.labels[(self.i - batch_size): self.i]

  def __iter__(self):
    for data, labels in zip(self.data, self.labels):
      yield data, labels

  def __len__(self):
    return len(self.data)

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def feature_size(self):
    return len(self.data[0]) if len(self.data) != 0 else 0

def split_interval(data, labels, ratios: [float]):
  total = len(data)
  intervals = []
  start = 0
  for i, ratio in enumerate(ratios):
    n_data = int(total * ratio)
    intervals.append([start, start + n_data])
    start += n_data

  if intervals[-1][1] != total:
    intervals[-1][1] = total

  return [Data(data[start:end], labels[start:end]) for start, end in intervals]


if __name__ == '__main__':
  def test():
    data = Data([1, 2, 3], [4, 5, 6])
    assert data.get(1) == ([1], [4])
    assert data.get(1) == ([2], [5])
    assert data.get(1) == ([3], [6])
  test()
