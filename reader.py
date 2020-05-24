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

def generate_SMA(data, target_index, days):
  sma = []
  cur_days = 0
  cur_sum = 0
  while cur_days < days and cur_days < len(data):
    cur_sum += data[cur_days][target_index]
    cur_days += 1
    sma.append(cur_sum / cur_days)

  while cur_days < len(data):
    cur_sum += data[cur_days][target_index] - data[cur_days - days][target_index]
    sma.append(cur_sum / days)
    cur_days += 1

  return sma

if __name__ == '__main__':
  def test():
    spy = SPY('./SPY.csv')
    data = [[i] for i in range(10)]
    labels = generate_SMA(data, 0, 2)
    assert labels[0] == 0
    for i in range(1, len(labels)):
      assert labels[i] == (data[i][0] + data[i-1][0]) * 0.5
  test()