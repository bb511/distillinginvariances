import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, num_conv_layers=2, temperature=1):
        super(SimpleCNN, self).__init__()
        self.conv_layers = self._make_conv_layers(in_channels, num_conv_layers)
        self.fc_layers = nn.Sequential(
            nn.Linear(9216, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
        self.temperature = temperature

    def _make_conv_layers(self, in_channels, num_conv_layers):    
        layers = []
        for i in range(num_conv_layers):
            # Adjust the number of output channels based on your requirements
            out_channels = 32 * (2**i)
            layers += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=7, stride=1, padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=7, stride=1, padding=1),
            ]
            in_channels = out_channels

        return nn.Sequential(*layers)
    
    """
    increase kernel size or 
    """

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        # Apply temperature scaling in the last layer
        x = x / self.temperature
        return x


class LeNet(nn.Module):
    def __init__(self, temperature):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.temperature = temperature

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # Apply temperature scaling in the last layer
        x = x / self.temperature
        return x
