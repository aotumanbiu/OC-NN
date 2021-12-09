# encoding: utf-8
import os
import glob
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm


class Cad(Dataset):
    def __init__(self, root, subFold, transform=None):
        super(Cad, self).__init__()
        self.path = os.path.join(root, subFold)
        if not os.path.exists(self.path):
            raise NotADirectoryError("文件路径不存在！！！！！！")
        self.transform = transform
        self.imgFiles = sorted(glob.glob(os.path.join(root, subFold) + "/*.jpg"))
        self.len = len(self.imgFiles)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        index = item % self.len
        img = Image.open(self.imgFiles[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img


def addNoise(inputs, sigma=0.1):
    noise_shape = inputs.shape
    noise = torch.normal(0, sigma, noise_shape)
    if inputs.is_cuda:
        outputs = inputs + noise.cuda()
    else:
        outputs = inputs + noise
    return outputs


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def CM(num_classes=2):
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)

    return confusion


def train_one_epoch(model, criterion, optimizer, train_loader, train_loader_lens, epoch, total_epoch, Cuda):
    tqdm_iter_train = tqdm(enumerate(train_loader), total=train_loader_lens)
    tqdm_iter_train.set_description(f"Trian_Epoch [{epoch + 1}/{total_epoch}]")

    # accu_num 为预测正确的样本数量
    accu_num = torch.zeros(1).cuda()
    total_loss = 0.0
    sample_num = 0

    # 训练
    model.train()
    for idx, data in tqdm_iter_train:

        with torch.no_grad():
            if Cuda:
                ins = addNoise(data[0].cuda())
                ins_labels = data[1].cuda()
            else:
                ins = addNoise(data[0])
                ins_labels = data[1]

        # 因为会在全连接处cat噪声, 所以样本数量为2倍
        sample_num += ins.shape[0] * 2

        pred, ous_labels = model((ins, ins_labels, True))

        optimizer.zero_grad()
        loss = criterion(pred, ous_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, ous_labels).sum()

        tqdm_iter_train.set_postfix(**{'loss': loss.item(),
                                       'lr': optimizer.param_groups[0]['lr'],
                                       'acc': accu_num.item() / sample_num})

    if (epoch + 1) > 0:
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(), './weights/Epoch%d-Total_Loss%.4f.pth' % (epoch + 1, total_loss))


def eval_one_epoch(model, test_loader, test_loader_lens, epoch, total_epoch, cm, Cuda):
    model.eval()
    tqdm_iter_test = tqdm(enumerate(test_loader), total=test_loader_lens)
    tqdm_iter_test.set_description(f"Test_Epoch [{epoch + 1}/{total_epoch}]")
    with torch.no_grad():
        for _, (val_images, val_labels) in tqdm_iter_test:

            if Cuda:
                val_images = val_labels.to("cuda")
            outputs, _ = model((addNoise(val_images), False))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            cm.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    cm.summary()


if __name__ == "__main__":
    a = torch.randn((3, 3)).cuda()
    addNoise(a)
