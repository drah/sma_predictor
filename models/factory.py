from . import models

def get_model(name):
  return {
    'NNR': models.NNRegression,
    'NNR2': models.NNRegression2,
  }[name]
