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
class VGGYoloV1(nn.Module):
    # Constructor
    def __init__(self, **kwargs):
        super(VGGYoloV1, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True)
        list(self.vgg.features.children())[0].requires_grad_(False) # freeze first convolution layer
        self.vgg.avgpool = None
        self.vgg.classifier = None

        self.conv1 = self._create_conv1()
        self.conv2 = self._create_conv2()
        self.fc = self._create_fcs(**kwargs)


    def forward(self, x):
        x = self.vgg.features(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x
    

    def _create_conv1(self):
        '''
        512x14x14 -> reduce to 7x7x1024 at last convolution
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.LeakyReLU(0.1)
        )


    def _create_conv2(self):
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
    model = VGGYoloV1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    # test 2 samples

    torchsummary.summary(model, input_size=(3, 448, 448))

    X = torch.randn(2, 3, 448, 448).to(DEVICE)
    print(model(X).shape)


if __name__=="__main__":
    # expect torch.Size([2, 1470])
    test() 
