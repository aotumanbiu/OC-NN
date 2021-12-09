# encoding: utf-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils import train_one_epoch
from model import OCN
import argparse


def main(args):
    # 进一步判断 GPU 是否可用
    if not torch.cuda.is_available():
        print("CUDA 不可用, 正在用 CUP 执行！！！！")
        args.cuda = False

    # --------------------------------------------------------------------------------------- #
    # 数据集加载
    # --------------------------------------------------------------------------------------- #
    Transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(root="./data/cat_dog/train/", transform=Transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    lens_train = len(train_loader)

    # ----------------------------------------------------------------------------------------------- #
    # 这里主要是想看每个epoch对应的测试集的准确率
    # ----------------------------------------------------------------------------------------------- #
    # test_dataset = datasets.ImageFolder(root="./data/cat_dog/test", transform=Transform)
    # test_loader = DataLoader(test_dataset, batch_size=batchSize * 2, shuffle=True, drop_last=True)
    # lens_test = len(test_loader)

    # ----------------------------------------------------------- #
    # 初始化模型
    # ----------------------------------------------------------- #
    model = OCN(num_classes=2, batchSize=args.batch_size, isTrain=True)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # ------------------------------------------ #
    # 只训练分类器部分
    # ------------------------------------------ #
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    for epoch in range(args.epoch):
        train_one_epoch(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        train_loader_lens=lens_train,
                        epoch=epoch,
                        total_epoch=args.epoch,
                        Cuda=args.cuda)

        lr_scheduler.step()

        # cm = CM()
        # eval_one_epoch(model=model,
        #                test_loader=test_loader,
        #                test_loader_lens=lens_test,
        #                epoch=epoch,
        #                total_epoch=100,
        #                cm=cm,
        #                Cuda=args.cuda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument("--cuda", metavar="True", type=bool, default=True,
                        help="change to False if you want to train on CPU (Seriously??)")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch_Size")
    parser.add_argument("--dataSet", type=str, default="./data/cat_dog/train", help="Dataset storage directory")
    parser.add_argument('--epoch', type=int, default=5, help="Total number of training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Adam: learning rate")
    args = parser.parse_args()

    # 开始训练
    main(args)
