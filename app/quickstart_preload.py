import torch
import time
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import pdb
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
# Download training data from open datasets.
set_train_args = {"train": True, "test": False}
set_loader_dict = {}
for set, train_arg in set_train_args.items():
    cpu_dataset = datasets.FashionMNIST(
        root="data",
        train=train_arg,
        download=True,
        transform=ToTensor(),
    )
    data_types = ("data", "targets")
    dev_tensors = {}
    dl = DataLoader(cpu_dataset, batch_size=len(cpu_dataset))
    for data_tup in dl:
        for data_type, data_val in zip(data_types, data_tup):
            dev_tensors[data_type] = data_val.to(device)
    dataset_args = [dev_tensors[k] for k in data_types]
    dev_dataset = TensorDataset(*dataset_args)
    set_loader_dict[set] = DataLoader(dev_dataset, batch_size=64, pin_memory=True)

for X, y in set_loader_dict["test"]:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

for epoch in range(5):
    before_train_time = time.time()
    train(set_loader_dict["train"], model, loss_fn, optimizer)
    after_train_time = time.time()
    train_time = after_train_time - before_train_time
    test(set_loader_dict["test"], model, loss_fn)
    test_time = time.time() - after_train_time
    print(f"epoch={epoch} times in seconds train={train_time:.1f} test={test_time:.1f}")

