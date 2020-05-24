
def handle_features(data):
  for features in data:
    features.pop(0)
    for i in range(len(features)):
      features[i] = float(features[i])

if __name__ == '__main__':
  pass