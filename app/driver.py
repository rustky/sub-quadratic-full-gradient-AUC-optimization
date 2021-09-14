from load_data import  load_data
from convolutional_NN import train_classifier
from convolutional_NN import test_classifier
from functional_square_loss import functional_square_loss


def main():
    trainloader, testloader = load_data()
    # train_classifier(trainloader, functional_square_loss)
    test_classifier(testloader)


if __name__ == '__main__':
    main()
