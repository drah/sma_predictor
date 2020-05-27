import numpy as np

def handle_features(data):
  for features in data:
    features.pop(0)
    for i in range(len(features)):
      features[i] = float(features[i])

def get_stat(data):
  data = np.array(data)
  return {
      'min': np.min(data, 0),
      'mean': np.mean(data, 0),
      'std': np.std(data, 0),
      'max': np.max(data, 0)}

def standardize(data, mean, std):
  for features in data:
    for i in range(len(features)):
      features[i] = (features[i] - mean[i]) / std[i]

if __name__ == '__main__':
  def test():
    import reader
    import data_pipeline
    from pprint import pprint
    spy = reader.SPY('./SPY.csv')
    labels = reader.generate_SMA(spy.data, 4, 30)
    train, val, test = data_pipeline.split_interval(
        spy.data, labels, [0.7, 0.15, 0.15])
    handle_features(train.data)
    stat = get_stat(train.data)
    standardize(train.data, stat['mean'], stat['std'])
    pprint(train.data[:5])
    pprint(get_stat(train.labels))

  test()
