import torch.nn as nn
import torch
#import deepspeed


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        hidden_layers: int,
        device: str,
        activation: str = "ReLU",
        from_saved_state_dict: str = None,
    ):
        super(MLP, self).__init__()

        self.device = device
        layers = []
        layers_sizes = [(input_dim, hidden_size)] + [
            (hidden_size, hidden_size)
        ] * hidden_layers

        for _, (input_size, output_size) in enumerate(layers_sizes):
            layers.append(nn.Linear(input_size, output_size))
            if activation == "Tanh":
                layers.append(nn.Tanh())
            elif activation == "Sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "ReLU":
                layers.append(nn.ReLU())
            else:
                raise Warning(
                    "The value passed as activation type is not none, but no activation with that name was "
                    "found!"
                )
        layers.append(nn.Linear(hidden_size, output_dim))
        # layers.append(nn.Softmax())

        self.layers = nn.Sequential(*layers).to(device)

        if from_saved_state_dict is not None:
            state_dict = torch.load(from_saved_state_dict)
            self.load_state_dict(state_dict)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        assert type(s) == torch.Tensor
        return self.layers(s)

    def training_loop(
        self,
        optimizer,
        criterion,
        num_epochs,
        train_loader,
        save_path_folder: str = "saved_models",
    ):
        #prof = deepspeed.profiling.flops_profiler.FlopsProfiler(model)
        profile_epoch = 0
        for epoch in range(num_epochs):
            #if epoch == profile_epoch:
                #prof.start_profile()
            for i, (x, labels) in enumerate(train_loader):
                x = x.view(-1, 784)
                outputs = self.forward(x.to(self.device))
                loss = criterion(outputs, labels.to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                    )
                #if epoch == profile_epoch and i == 0:
                    #prof.stop_profile()
                    #flops = prof.get_total_flops()
                    #prof.end_profile()
        # Save the trained model
        save_path = save_path_folder + "\mlp"
        torch.save(self.state_dict(), save_path)
        print(f"Model saved as {save_path}!")
        #print("Model flops: ", flops)

    def eval_loop(self, test_loader):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, 784)
                outputs = self.forward(images.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum().item()

            accuracy = correct / total
            print(f"Test Accuracy: {accuracy:.4f}")
