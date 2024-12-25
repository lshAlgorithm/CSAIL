import os
import torch
import torch.nn as nn
import time
from generate_pth import vgg19
from generate_pth import load_image

os.putenv('MLU_VISIBLE_DEVICES', '')
cfgs = [64, 'R', 64, 'R', 'M', 128, 'R', 128, 'R', 'M',
        256, 'R', 256, 'R', 256, 'R', 256, 'R', 'M',
        512, 'R', 512, 'R', 512, 'R', 512, 'R', 'M',
        512, 'R', 512, 'R', 512, 'R', 512, 'R', 'M']

IMAGE_PATH = 'data/strawberries.jpg'
VGG_PATH = 'models/vgg19.pth'


if __name__ == '__main__':
    input_image = load_image(IMAGE_PATH)
    # TODO: 生成VGG19网络模型并保存在net中
    net = vgg19()
    # TODO: 加载网络参数到net中
    net.load_state_dict(torch.load(VGG_PATH))
    # TODO: 模型进入推理模式
    net.eval()
    st = time.time()
    # TODO: 计算 net,得到prob
    prob = net(input_image)
    print("cpu infer time:{:.3f} s".format(time.time()-st))
    with open('./labels/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        _, indices = torch.sort(prob, descending=True)
    print("Classification result: id = %s, prob = %f " %
          (classes[indices[0][0]], prob[0][indices[0][0]].item()))
    if classes[indices[0][0]] == 'strawberry':
        print('TEST RESULT PASS.')
    else:
        print('TEST RESULT FAILED.')
        exit()
