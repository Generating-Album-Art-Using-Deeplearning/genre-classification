import torch.nn.functional as F
from torch import nn

'''
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d((2,2))
        self.fc = nn.Sequential(
            nn.Linear(89856, 5000),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(5000, 255),
            nn.ReLU(),
            nn.Dropout(),
        )
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(10, -1)
        x = self.fc(x)
        x = self.fc2(x)
        
        return x
'''

class W2VModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding):
        super(W2VModel, self).__init__()
        assert (
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.model = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(stride, filter_size, padding)):
            self.model.add_module(
                "model_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout()
                )
            )
            input_dim = hidden_dim

        self.model_2 = nn.Sequential()
        self.model_2.add_module(
            "fc_layer",
            nn.Sequential(
                nn.Linear(6669, 500),
                nn.ReLU(),
                nn.Linear(500, 255),
                nn.ReLU()
            )
        )
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1,6669)
        x = self.model_2(x)
        return x