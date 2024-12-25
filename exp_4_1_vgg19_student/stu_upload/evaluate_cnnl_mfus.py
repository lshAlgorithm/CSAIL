import os
import torch
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch.nn as nn
import time
from PIL import Image
from torchvision import transforms
from generate_pth import vgg19
from generate_pth import load_image

torch.set_grad_enabled(False)
ct.set_device(0)
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
    example_forward_input = torch.rand((1, 3, 224, 224), dtype=torch.float)
    # TODO: 使用JIT对模型进行trace，把动态图转化为静态图，得到net_trace
    # convert into TorchScript
    net_trace = torch.jit.trace(net, example_forward_input)
    # TODO: 将输入图像拷贝到MLU设备
    input_image_mlu = input_image.to(ct.mlu_device())
    # TODO: 将net_trace拷贝到MLU设备
    net_trace_mlu = net_trace.to(ct.mlu_device())
    st = time.time()
    # TODO: 进行推理，得到prob
    prob = net_trace_mlu(input_image_mlu)
    print("mlu370<cnnl backend> infer time:{:.3f} s".format(time.time()-st))
    # TODO: 将prob从MLU设备拷贝到CPU设备
    prob = prob.to('cpu')
    with open('./labels/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        _, indices = torch.sort(prob, descending=True)
    print("Classification result: id = %s, prob = %f " %
          (classes[indices[0][0]], prob[0][indices[0][0]].item()))
    if classes[indices[0][0]] == 'strawberry':
        print('TEST RESULT PASS.')
    else:
        print('TEST RESULT FAILED.')
