import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from config import cfg

#
#   We use the pre-trained ResNet50, removing the usual top layer and using
#    a new Linear with the right amount of classes of output.
#   The model is placed on the correct selected device for efficiency.
#
class Custom_ResNet():
    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 4)
        self.model = self.model.to(cfg.device)

    def Model(self):
        return self.model

custom_resnet = Custom_ResNet().Model()