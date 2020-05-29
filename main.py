import argparse
import data_pipeline
import features
import models

import reader

def main(args):
  spy = reader.SPY(args.data_path)
  labels = reader.generate_SMA(spy.data, args.label_index, args.sma_days)
  train, val, test = data_pipeline.split_interval(
      spy.data, labels, [args.train_ratio, args.val_ratio, args.test_ratio])

  stat = features.handle_features(train.data)
  features.handle_features(val.data, stat['mean'], stat['std'])
  features.handle_features(test.data, stat['mean'], stat['std'])

  model = models.factory.get_model(args.model)(train.feature_size, args.ckpt, args.save_dir)
  model.init_session()
  if args.phase == 1:
    model.fit(
        train=train,
        val=val,
        start_step=args.start_step,
        end_step=args.end_step,
        batch_size=args.batch_size,
        val_step=args.val_step,
        val_batch_size=args.val_batch_size,
        es_tolerance=args.es_tolerance)
    model.evaluation(test)
  elif args.phase == 0:
    model.evaluation(test)
  elif args.phase == 2:
    model.predict(test)

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
  parser.add_argument('--save_dir', dest='save_dir', type=str, default='',
      help='the directory for saving training logs and checkpoints')
  parser.add_argument('--start_step', dest='start_step', type=int, default=0,
      help='the number from which training step starts')
  parser.add_argument('--end_step', dest='end_step', type=int, default=100000,
      help='the number from which training step ends')
  parser.add_argument('--batch_size', dest='batch_size', type=int, default=100,
      help='the size of each batch for training')
  parser.add_argument('--val_step', dest='val_step', type=int, default=100,
      help='the number of steps for validation once')
  parser.add_argument('--val_batch_size', dest='val_batch_size', type=int, default=100,
      help='the size of each batch for validation')
  parser.add_argument('--es_tolerance', dest='es_tolerance', type=int, default=100,
      help='early stopping tolerance')
  parser.add_argument('--phase', dest='phase', type=int, default=1,
      help='1: fit and evaluation, 0: evaluation')

  args = parser.parse_args()
  if not args.save_dir:
    args.save_dir = args.model + '_logs'
  main(args)