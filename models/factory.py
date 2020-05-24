from . import models

def get_model(name):
  return {
    'NNR': models.NNRegression,
  }[name]
