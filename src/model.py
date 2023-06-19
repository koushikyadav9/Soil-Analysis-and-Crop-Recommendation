import torch.nn as nn

import torchvision.models as models
# class ClassifierModel(nn.Module):
#     def __init__(self):
#         super(ClassifierModel, self).__init__()
        
#         # Define the convolutional layers
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Define the fully connected layers
#         self.fc1 = nn.Linear(32 * 56 * 56, 128)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.fc2(x)
#         # x = self.softmax(x)
#         return x


class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        
        # Load the pre-trained MobileNetV2 model
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Modify the last fully connected layer for classification
        num_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(num_features, 5)
    
    def forward(self, x):
        x = self.mobilenet(x)
        return x