import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader,Dataset

from model import MobileNetV2

# 自定义数据集FlowerData
# 读取的数据目录结构：
"""
            directory/
            ├── class_x
            │   ├── xxx.jpg
            │   ├── yyy.jpg
            │   └── ...   
            └── class_y
                ├── 123.jpg
                ├── 456.jpg
                └── ...
"""
class FlowerData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        classes = sorted(entry.name for entry in os.scandir(self.root_dir) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.images = self.get_images(self.root_dir, self.class_to_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        path, target = self.images[index]
        with open(path, "rb") as f:
            img = Image.open(f)
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)   #对样本进行变换

        return image,target

    def get_images(self, directory, class_to_idx):
        images = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    images.append(item)

        return images

# 训练和评估
def fit(epochs, model, loss_function, optimizer, train_loader, validate_loader, device):
    t0 = time.time()
    best_acc = 0.0
    save_path = './MobileNetV2.pth'
    train_steps = len(train_loader)
    model.to(device)
    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, total=train_steps) # 进度条
        for step, (images, labels) in enumerate(train_bar):
            optimizer.zero_grad() # grad zero 
            logits = model(images.to(device)) # Forward
            loss = loss_function(logits, labels.to(device)) # loss
            loss.backward() # Backward
            optimizer.step() # optimizer.step

            _, predict = torch.max(logits, 1)
            train_acc += torch.sum(predict == labels.to(device))
            
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)

        train_accurate = train_acc / len(train_loader.dataset)

        # 验证
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, total=len(validate_loader)) # 进度条
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))

                _, val_predict = torch.max(outputs, 1)
                val_acc += torch.sum(val_predict == val_labels.to(device))

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
        val_accurate = val_acc / len(validate_loader.dataset)

        print('[epoch %d] train_loss: %.3f - train_accuracy: %.3f - val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, val_accurate))

        # 保存最好的模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print("\n{} epochs completed in {:.0f}m {:.0f}s.".format(epochs,(time.time() - t0) // 60, (time.time() - t0) % 60))


def main():
    # 有GPU，就使用GPU训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 超参数
    batch_size = 32
    epochs = 10
    learning_rate = 0.0001

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 初始化自定义FlowerData类，设置数据集所在路径以及变换
    flower_data = FlowerData('../flower_photos',transform=data_transform)
    print("Dataset class: {}".format(flower_data.class_to_idx))

    # 数据集随机划分训练集（80%）和验证集（20%）
    train_size = int(len(flower_data) * 0.8)
    validate_size = len(flower_data) - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(flower_data, [train_size, validate_size])
    print("using {} images for training, {} images for validation.".format(len(train_dataset),len(validate_dataset)))

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process \n'.format(nw))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=True, num_workers=nw)

    # 实例化模型，设置类别个数num_classes
    net = MobileNetV2(num_classes=5).to(device)

    # 使用预训练权重 https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./mobilenet_v2-b0353104.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)

    pre_weights = torch.load(model_weight_path, map_location=device)
    # print("The type is:".format(type(pre_weights)))

    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # 通过requires_grad == False的方式来冻结特征提取层权重，仅训练后面的池化和classifier层
    for param in net.features.parameters():
        param.requires_grad = False

    # 使用交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()

    # 使用adam优化器, 仅仅对最后池化和classifier层进行优化
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate)

    # 输出网络结构
    #print(summary(net, (3, 224, 224)))

    # 训练和验证模型
    fit(epochs, net, loss_function, optimizer, train_loader, validate_loader, device)

if __name__ == '__main__':
    main()
