import sys
train_args = sys.argv[1:]
from convolutional_NN import train_classifier
train_classifier(*train_args)
