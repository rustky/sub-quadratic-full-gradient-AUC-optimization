import pandas as pd
from load_data import load_data
from convolutional_NN import train_classifier
from functional_square_loss import functional_square_loss
from plotnine import ggplot, geom_line, aes, labs
from functional_square_hinge_loss import functional_square_hinge_loss
from naive_square_loss import naive_square_loss
from naive_square_hinge_loss import naive_square_hinge_loss


def main():
    SEED = 123
    imratio = 0.5
    lr = 1e-06
    num_epochs = 25
    trainloader, testloader = load_data(SEED, imratio)
    train_results = train_classifier(trainloader, testloader, functional_square_hinge_loss, num_epochs, lr)
    train_auc_dict = {}
    for results_length in range(num_epochs):
        train_auc_dict["auc"] = train_results[results_length]['train_auc']
        train_auc_dict["epochs"] = train_results[results_length]['epoch'] + 1
    train_auc_df = pd.DataFrame(train_auc_dict, index=[0])
    ggplot(data= train_auc_df) + aes(x="epochs", y="auc") + geom_line() + labs(title="Train AUC vs Epochs")


if __name__ == '__main__':
    main()
