import pandas as pd
from load_data import load_data
from convolutional_NN import train_classifier
from functional_square_loss import functional_square_loss
from functional_square_hinge_loss import functional_square_hinge_loss
from naive_square_loss import naive_square_loss
from naive_square_hinge_loss import naive_square_hinge_loss
import time
import numpy as np
import matplotlib.pyplot as plt

def signal_handler(signum, frame):
    raise Exception("Timed out!")


def main():
    SEED = 123
    imratio = 0.5
    lr = .5e-06
    num_epochs = 10
    batch_size = 500
    use_subset = False
    algo_list = [functional_square_loss, functional_square_hinge_loss]
    str_algo_list = ['functional_square_loss', 'functional_square_hinge_loss']
    train_auc_df = pd.DataFrame()

    trainloader, testloader = load_data(SEED, use_subset, batch_size, imratio)
    for x in range(2):
        train_auc_list = train_classifier(trainloader, testloader, algo_list[x], num_epochs, lr)
        train_auc_df[str_algo_list[x]] = train_auc_list
    print(train_auc_df)


if __name__ == '__main__':
    main()
