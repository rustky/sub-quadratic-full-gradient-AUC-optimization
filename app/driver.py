from load_data import  load_data
from convolutional_NN import train_classifier
from convolutional_NN import test_classifier
if __name__ == '__main__':
    trainloader, trainset, testloader, testset = load_data()
    train_classifier(trainloader, trainset)
    # test_classifier(testloader, testset)
