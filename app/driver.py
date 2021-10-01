from load_data import load_data
from convolutional_NN import train_classifier
from functional_square_loss import functional_square_loss
from functional_square_hinge_loss import functional_square_hinge_loss
import torch
import torchvision
for mod_name in 'torch', 'torchvision':
    print(mod_name, eval(mod_name).__version__)
    
def main():
    SEED = 123
    imratio = 0.5
    lr = .5e-06
    num_epochs = 10
    batch_size = 500
    use_subset = True
    algo_list = [functional_square_loss, functional_square_hinge_loss]
    trainloader, testloader = load_data(
        SEED, use_subset, batch_size, imratio)
    for algo in algo_list:
        train_classifier(
            trainloader, testloader, algo, num_epochs, lr)

if __name__ == '__main__':
    main()
