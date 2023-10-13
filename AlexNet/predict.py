import json
import os.path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_path = './pre.jpg'
    assert os.path.exists(img_path), f'file: {img_path} does not exist!'
    img = Image.open(img_path)

    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  # expand batch on the first dimension

    try:
        json_path = './class_indexes.json'
        assert os.path.exists(json_path), f'file: {img_path} does not exist!'
        with open('./class_indexes.json', 'r') as f:
            class_indict = json.load(f)
    except Exception as e:
        print(e)
        exit(-1)

    model = AlexNet(num_classes=5).to(device)
    weights_path = './AlexNet.pth'
    assert os.path.exists(weights_path), f'file: {weights_path} does not exist!'
    model.load_state_dict(torch.load(weights_path))  # 将加载的模型权重加载到模型对象中的状态字典中。模型的状态字典是一个字典对象，它将模型的参数名称映射到对应的权重张量

    model.eval()
    with torch.no_grad():
        # squeeze()去除输出张量中维度为1的维度，得到更简化的张量
        output = torch.squeeze(model(img))
        # 计算每个类别的概率分布，并得到一个新的张量 predict
        predict = torch.softmax(output, dim=0)  # dim=0 表示对第一个维度（索引为0）进行 softmax 操作，即在类别维度上进行操作
        # argmax()找到 predict 张量中概率最大的类别，并将其转换为NumPy数组形式
        predict_cla = torch.argmax(predict).numpy()
    # 打印最大概率的预测对应的标签以及概率值
    print(class_indict[str(predict_cla)], predict[predict_cla].item())
    plt.show()


if __name__ == '__main__':
    main()
