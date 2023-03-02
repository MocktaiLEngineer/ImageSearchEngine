import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class CNNModel(nn.Module):
    def __init__(self, embedding_size):
        super(CNNModel, self).__init__()
        resnet = models.resnet152(pretrained=True)
        module_list = list(resnet.children())[:-1] # exlude the last layer to get the embeddings
        self.resnet_module = nn.Sequential(*module_list)
        self.embedding_layer = nn.Linear(resnet.fc.in_features, embedding_size)
    
    def forward(self, input_images):
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)
        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
        embedding = self.embedding_layer(resnet_features)
        return embedding