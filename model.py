# encoding: utf-8
import torch
import torch.nn as nn
from torchvision import models


class OCN(nn.Module):
    def __init__(self, num_classes, batchSize=3):

        backbone = models.vgg16(pretrained=True)
        backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:2])

        super(OCN, self).__init__()

        self.num_classes = num_classes

        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Linear(4096, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, 128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, self.num_classes))

        # self.bn1d = nn.BatchNorm1d(1, affine=False)
        self.relu = nn.ReLU(inplace=True)

        self.labels = torch.ones((batchSize,)).long()

    def forward(self, inputs):
        labels = None
        ins_labels = None

        # train
        if len(inputs) == 3:
            x, ins_labels, mode = inputs
        # eval or test
        else:
            x, mode = inputs

        x = self.backbone(x)

        if mode:
            gaussian = torch.normal(0, 0.1, x.shape)
            if x.is_cuda:
                gaussian = gaussian.cuda()
                self.labels = self.labels.cuda()

            # 把正样本标签和负样本标签concat
            labels = torch.cat((ins_labels, self.labels), dim=0)
            # 打乱数据和标签顺序, 但保持对应不变
            x = torch.cat((x, gaussian), dim=0)
            x, labels = self.shuffle(x, labels)
            x = self.relu(x)

        x = self.classifier(x)

        return x, labels

    @staticmethod
    def shuffle(img, labels):
        shuffle = torch.randperm(img.shape[0])
        img = img[shuffle, ...]
        labels = labels[shuffle]
        return img, labels


if __name__ == "__main__":
    model = OCN(num_classes=2).cuda()
    ins = torch.randn((3, 3, 224, 224)).cuda()
    model(ins)
