import torch
import torch.nn as nn
import torchvision

import torchsummary


# ==========
# Setup
# ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ==========
# Model Definition
# ==========
class ResnetYoloV1(nn.Module):
    # Constructor
    def __init__(self, **kwargs):
        super(ResnetYoloV1, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1.requires_grad_(False) # freeze first convolution layer
        self.resnet.avgpool = None
        self.resnet.fc = None

        self.layer5 = self._create_layer5()
        self.layer6 = self._create_layer6()
        self.fc = self._create_fcs(**kwargs)
        

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)
        
        x = self.fc(x)
        return x
    

    def _create_layer5(self):
        '''
        512x14x14 -> reduce to 7x7x1024 at last convolution
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.LeakyReLU(0.1)
        )
    

    def _create_layer6(self):
        '''
        7x7x1024 -> 7x7x1024
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
    

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5)), # Shape: (S, S, 30) where C+B*5 = 30
        )


def test(S=7, B=2, C=20):
    model = ResnetYoloV1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    # test 2 samples

    torchsummary.summary(model, input_size=(3, 448, 448))

    X = torch.randn(2, 3, 448, 448).to(DEVICE)
    print(model(X).shape)


if __name__=="__main__":
    # expect torch.Size([2, 1470])
    test() 
