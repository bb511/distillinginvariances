from models.resnet import resnet18_mnist
from models.mlp import MLP
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import deepspeed

def main():
    calc_mlp(2048)
    calc_mlp(512)
    calc_cnn()

def calc_cnn():
    model_cnn = resnet18_mnist().to(device)
    criterion_cnn = torch.nn.CrossEntropyLoss()
    optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=lr)
    prof = deepspeed.profiling.flops_profiler.FlopsProfiler(model_cnn)
    profile_epoch = 0
    for epoch in range(1):
        if epoch == profile_epoch:
            prof.start_profile()
        for i, (images, labels) in enumerate(train_loader):
            outputs = model_cnn(images.to(device))
            loss = criterion_cnn(outputs, labels.to(device))
            optimizer_cnn.zero_grad()
            loss.backward()
            optimizer_cnn.step()
            if epoch == profile_epoch and i == 0:
                prof.stop_profile()
                flops = prof.get_total_flops(as_string=True)
                prof.end_profile()
                print("Model flops CNN: ", flops)
                return

        

def calc_mlp(hidden_size):
    model_mlp = MLP(input_dim = 784, output_dim= num_classes, hidden_size= hidden_size,
                            hidden_layers= 4, device=device)
    criterion_mlp = torch.nn.CrossEntropyLoss()
    optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=lr)
    prof = deepspeed.profiling.flops_profiler.FlopsProfiler(model_mlp)
    profile_epoch = 0
    for epoch in range(1):
        if epoch == profile_epoch:
            prof.start_profile()
        for i, (x, labels) in enumerate(train_loader):
            x = x.view(-1, 784)
            outputs = model_mlp.forward(x.to(device))
            loss = criterion_mlp(outputs, labels.to(device))

            optimizer_mlp.zero_grad()
            loss.backward()
            optimizer_mlp.step()
            if epoch == profile_epoch and i == 0:
                prof.stop_profile()
                flops = prof.get_total_flops(as_string=True)
                prof.end_profile()
                temp = "Model flops MLP_" + str(hidden_size)
                print(temp, flops)
                return
    

if __name__ == "__main__":
    device = "cuda"
    num_classes = 10
    lr = 0.001
    batch_size = 64

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    main()