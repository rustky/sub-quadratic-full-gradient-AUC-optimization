import torch
import numpy as np
import functional_loss_test as test
import naive_square_loss
import naive_square_hinge_loss
SEED = 1
for i in range(10):
    torch.manual_seed(SEED)
    size = 10
    x = np.array([0, 1])
    predictions = torch.rand(size)
    labels = torch.from_numpy(np.repeat(x, size/2))
    function_hinge = test.square_hinge_test(predictions, labels)
    naive_square = naive_square_loss.naive_square_loss(predictions, labels, 1)
    naive_hinge = naive_square_hinge_loss.naive_square_hinge_loss(predictions, labels)
    functional_square = test.square_test(predictions, labels)
    print("Functional Hinge: ", function_hinge)
    print("Naive Hinge: ", naive_hinge)
    print("Naive Square: ", naive_square)
    print("Functional: ", functional_square)
    SEED += 2**i