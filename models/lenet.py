from torch import nn

class LeNet(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(1024, 128)
#         self.fc2 = nn.Linear(128, 10)
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        in_size = x.size(0)

        x = self.conv1(x)
        x = self.act(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.mp(x)

        x = x.view(in_size, -1)
        # print('fc1', x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x