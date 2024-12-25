import time
from torchvision.utils import save_image
from torchvision.models import vgg19
import torch
from torch.utils.data import DataLoader
import torch_mlu

from utils import COCODataSet, TransNet


if __name__ == '__main__':
    # TODO: 使用cpu生成图像转换网络模型并保存在g_net中
    g_net = TransNet().to('cpu')
    # TODO: 从/models文件夹下加载网络参数到g_net中
    g_net.load_state_dict(torch.load('models/fst.pth', map_location='cpu'))
    print("g_net build PASS!\n")
    # TODO：将g_net模型转化为eval,并转化为浮点类型，输出得到net
    net = g_net.eval().float()
    data_set = COCODataSet()
    print("load COCODataSet PASS!\n")
    batch_size = 1
    data_group = DataLoader(data_set, batch_size, True, drop_last=True)
    example_forward_input = torch.rand((1, 3, 512, 512), dtype=torch.float)
    # TODO: 使用JIT对net模型进行trace，得到net_trace
    net_trace = torch.jit.trace(net, example_forward_input)
    for i, image in enumerate(data_group):
        print(f"The {i} image will be predicted.")
        image_c = image.cpu()
        # 将image_c图片拷贝到MLU设备，得到input_image_c
        input_image_c = image_c.to(torch_mlu.core.mlu_model.get_device())
        # 将net_trace模型拷贝到MLU设备，得到net_mlu
        net_mlu = net_trace.to(torch_mlu.core.mlu_model.get_device())
        start = time.time()
        # TODO: 对input_image_c计算 net_mlu,得到image_g_mlu
        image_g_mlu = net_mlu(input_image_c)
        image_g_mlu = image_g_mlu.cpu()
        end = time.time()
        delta_time = end - start
        print("Inference (mfus) processing time: %s" % delta_time)
        # TODO: 利用save_image函数将tensor形式的生成图像image_g_mlu以及输入图像image_c以jpg格式左右拼接的形式保存在/out/mlu_cnnl_mfus/文件夹下
        save_image(torch.cat((image_c, image_g_mlu), -1),
                   f'out/mlu_cnnl_mfus/#{i}_result.jpg')
    print("TEST RESULT PASS!\n")
