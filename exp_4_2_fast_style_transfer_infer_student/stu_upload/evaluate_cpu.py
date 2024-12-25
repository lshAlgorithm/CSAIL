import time
from torchvision.utils import save_image
from torchvision.models import vgg19
import torch
from torch.utils.data import DataLoader

from utils import COCODataSet, TransNet

if __name__ == '__main__':
    # TODO: 使用cpu生成图像转换网络模型并保存在g_net中
    g_net = TransNet().to('cpu')
    # TODO:从/models文件夹下加载网络参数到g_net中
    pt_param = torch.load('models/fst.pth', map_location='cpu')
    g_net.load_state_dict(pt_param)
    print("g_net build  PASS!\n")
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")

    batch_size = 1
    data_group = DataLoader(data_set, batch_size, True, drop_last=True)

    for i, image in enumerate(data_group):
        image_c = image.cpu()
        # print(image_c.shape)
        start = time.time()
        # TODO: 计算 g_net,得到image_g
        image_g = g_net(image_c)
        end = time.time()
        delta_time = end - start
        print("Inference (CPU) processing time: %s" % delta_time)
        # TODO: 利用save_image函数将tensor形式的生成图像image_g以及输入图像image_c以jpg格式左右拼接的形式保存在/out/cpu/文件夹下
        save_image(torch.cat((image_c, image_g), -1),
                   f'out/cpu/#{i}_result.jpg')
    print("TEST RESULT PASS!\n")
