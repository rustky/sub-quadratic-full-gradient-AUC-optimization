from convolutional_NN import train_classifier
from functional_square_loss import functional_square_loss
from functional_square_hinge_loss import functional_square_hinge_loss
import torch
import torchvision
for mod_name in 'torch', 'torchvision':
    print(mod_name, eval(mod_name).__version__)
    
if __name__ == '__main__':
    import sys
    prog, imratio, lr, num_epochs, batch_size = sys.argv
    # imratio = 0.5
    # lr = .5e-06
    # num_epochs = 10
    # batch_size = 500
    loss_list = [functional_square_hinge_loss, functional_square_loss]
    for loss in loss_list:
        train_classifier(
            int(batch_size),
            float(imratio),
            loss,
            int(num_epochs),
            float(lr))
