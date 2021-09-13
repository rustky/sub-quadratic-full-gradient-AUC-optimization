from load_data import  load_data
from convolutional_NN import train_classifier
from convolutional_NN import test_classifier


def main():
    trainloader, testloader = load_data()
    train_classifier(trainloader)
    # test_classifier(testloader)


if __name__ == '__main__':
    main()
