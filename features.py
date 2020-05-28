import numpy as np

def handle_features(data, mean=None, std=None):
  months = []
  days = []
  for features in data:
    date = features[0].split('-')
    features[0] = float(date[0])
    months.append(date[1])
    days.append(date[2])

  stat = get_stat(data)
  mean = mean if mean is not None else stat['mean']
  std = std if std is not None else stat['std']
  for month, day, features in zip(months, days, data):
    for i in range(len(features)):
      features[i] = float(features[i])
    _standardize(features, mean, std)
    features.extend(one_hot(int(month)-1, 12))
    features.extend(one_hot(int(day)-1, 31))
  return stat

def get_stat(data):
  data = np.array(data)
  return {
      'min': np.min(data, 0),
      'mean': np.mean(data, 0),
      'std': np.std(data, 0),
      'max': np.max(data, 0)}

def standardize(data, mean, std):
  for features in data:
    _standardize(features, mean, std)

def _standardize(features, mean, std):
  for i in range(len(features)):
    features[i] = (features[i] - mean[i]) / std[i]

def one_hot(data, depth):
  return [0 if i != data else 1 for i in range(depth)]

if __name__ == '__main__':
  def test():
    import reader
    import data_pipeline
    from pprint import pprint
    spy = reader.SPY('./SPY.csv')
    labels = reader.generate_SMA(spy.data, 4, 30)
    train, val, test = data_pipeline.split_interval(
        spy.data, labels, [0.7, 0.15, 0.15])
    pprint(train.data[:5])
    handle_features(train.data)
    pprint(train.data[:5])
    pprint(get_stat(train.labels))

  test()
