import os
import json
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils import Cad, addNoise
from tqdm import tqdm
from model import OCN
from utils import ConfusionMatrix
import time

# -------------------------------- #
# 判断 CUDA 是否可用
# -------------------------------- #
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# --------------------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------------------- #
Transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

# dataset = Cad(root="./data/cat_dog/test", subFold="cats", transform=Transform)
dataset = datasets.ImageFolder(root="./data/cat_dog/test", transform=Transform)
test_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
lens = len(test_loader)

# ------------------------------------------------------------------------------------- #
# 初始化混淆矩阵
# ------------------------------------------------------------------------------------- #
json_label_path = './class_indices.json'
assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
json_file = open(json_label_path, 'r')
class_indict = json.load(json_file)

labels = [label for _, label in class_indict.items()]
confusion = ConfusionMatrix(num_classes=2, labels=labels)

# --------------------------------------------------------------------------- #
# 建立模型, 加载权重
# --------------------------------------------------------------------------- #
model = OCN(num_classes=2, isTrain=False).to(device)
print("开始加载权重！！！！")
state_dict = torch.load("Epoch4-Total_Loss19.8168.pth", map_location=device)
model.load_state_dict(state_dict)
print("权重加载完成！！！！")

# ------------------------------------------------- #
# 测试
# ------------------------------------------------- #
model.eval()
with torch.no_grad():
    star = time.time()
    for val_data in tqdm(test_loader):
        val_images, val_labels = val_data
        outputs, _ = model((val_images.to(device), False))
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
confusion.plot()
confusion.summary()
