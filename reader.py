
class SPY:
  def __init__(self, data_path):
    self.data_path = data_path
    self._read(data_path)
  
  def _read(self, data_path):
    examples = []
    with open(data_path, 'r') as f:
      header = f.readline()
      for line in f:
        line = line.strip().split(',')
        date = line[0:1]
        values = list(map(float, line[1:]))
        examples.append(date + values)

    self._header = header
    self._examples = examples

  @property
  def header(self):
    return self._header

  @property
  def data(self):
    return self._examples

class Spliter:
  def __init__(self):
    pass

  def split_interval(self, data, ratios: [float]):
    total = len(data)
    intervals = []
    start = 0
    for i, ratio in enumerate(ratios):
      n_data = int(total * ratio)
      intervals.append([start, start + n_data])
      start += n_data

    if intervals[-1][1] != total:
      intervals[-1][1] = total

    return [data[start:end] for start, end in intervals]

if __name__ == '__main__':
  spy = SPY('./SPY.csv')
  spliter = Spliter()
  train, val, test = spliter.split_interval(spy.data, [0.7, 0.15, 0.15])
  print(len(train), len(val), len(test))
  