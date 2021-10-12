# sub-quadratic-full-gradient-AUC-optimization
## 9/30/2021
./app folder
1. ```Driver.py``` to run the complete experiment 
   1. Command Line Args  ex. ```$ python3 driver.py 10 0.1 square_hinge .0001 ./results/test CAT_VS_DOG```
      1. Batch Size
      2. Imbalance Ratio
      3. Loss Function
      4. Learning Rate
      5. Output Directory
      6. Dataset (CIFAR10, CAT_VS_DOG, CIFAR100, CHEXPERT)
      7. Model 
2. ```Convolutional_NN.py``` to train and test the classifiers. Outputs a dataframe, with list of dictionaries per epoch
3. ```Load_data.py``` to create trainloader and testloader

Dependencies: LibAUC, Torch, Numpy, Pandas, functional_square_loss, and functional_square_hinge_loss
