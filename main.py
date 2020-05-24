import argparse
import data_pipeline
import feature_engineering
import models

import reader

def main(args):
  spy = reader.SPY(args.data_path)
  labels = reader.generate_SMA(spy.data, args.label_index, args.sma_days)
  train, val, test = data_pipeline.split_interval(
      spy.data, labels, [args.train_ratio, args.val_ratio, args.test_ratio])

  feature_engineering.handle_features(train.data)
  feature_engineering.handle_features(val.data)
  feature_engineering.handle_features(test.data)

  model = models.factory.get_model(args.model)(train.feature_size, args.ckpt, args.save_dir)
  model.fit(train.data, train.labels, val.data, val.labels)
  model.validate(test.data, test.labels)

if __name__ == '__main__':
  parser = argparse.ArgumentParser('sma_predictor')
  parser.add_argument('--data', dest='data_path', default='./SPY.csv',
      help='path to data')
  parser.add_argument('--label_index', dest='label_index', type=int, default=4,
      help='the column index of the close price')
  parser.add_argument('--sma_days', dest='sma_days', type=int, default=30,
      help='the number of days for calculating sma')
  parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=0.7,
      help='the ratio for splitting training data')
  parser.add_argument('--val_ratio', dest='val_ratio', type=float, default=0.15,
      help='the ratio for splitting validation data')
  parser.add_argument('--test_ratio', dest='test_ratio', type=float, default=0.15,
      help='the ratio for splitting test data')
  parser.add_argument('--model', dest='model', type=str, default='NNR',
      help='the name of the model')
  parser.add_argument('--ckpt', dest='ckpt', type=str, default=None,
      help='checkpoint of model')
  parser.add_argument('--save_dir', dest='save_dir', type=str, default='./NNR_logs',
      help='the directory for saving training logs and checkpoints')

  args = parser.parse_args()
  main(args)