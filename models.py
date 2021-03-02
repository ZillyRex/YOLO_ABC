import torch
import torch.nn as nn
import torchvision


class YOLOV0(nn.Module):
    def __init__(self, backbone, pretrained):
        super(YOLOV0, self).__init__()

        assert backbone in ['alexnet', 'vgg16']
        if backbone == 'alexnet':
            self.features = torchvision.models.alexnet(
                pretrained=pretrained).features

        elif backbone == 'vgg16':
            self.features = torchvision.models.vgg16(
                pretrained=pretrained).features

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        if backbone == 'alexnet':
            self.detector = nn.Sequential(
                nn.Linear(256*8*8, 4096),
                nn.BatchNorm1d(4096),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Linear(4096, 320),
                nn.Sigmoid()
            )
        elif backbone == 'vgg16':
            self.detector = nn.Sequential(
                nn.Linear(512*8*8, 4096),
                nn.BatchNorm1d(4096),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Linear(4096, 320),
                nn.Sigmoid()
            )

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 1)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.detector(x)
        b = x.size(0)
        x = x.view(b, 8, 8, 5)
        return x


def main():
    x = torch.randn(4, 3, 512, 512)
    yolov0 = YOLOV0(torch.device('cpu'))
    feature = yolov0(x)
    print(feature.shape)


if __name__ == '__main__':
    main()
