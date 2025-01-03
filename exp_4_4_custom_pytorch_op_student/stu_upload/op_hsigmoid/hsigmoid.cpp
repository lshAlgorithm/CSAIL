//Pytorch扩展头文件的引用
#include <torch/extension.h> 
using namespace std; 

//hsigmoid_cpu函数的具体实现
torch::Tensor hsigmoid_cpu(const torch::Tensor & dets) {
  //TODO: 将输入的tensor转化为浮点类型的vector
  auto input = dets.accessor<float, 2>();
  int batch_size = input.size(0);
  int input_size = input.size(1); 
  //TODO: 创建一个浮点类型的output_data，output_data为大小与输入相同的vector
  vector<float> output_data(batch_size * input_size);
  //TODO: 对于输入向量的每个元素计算hsigmoid
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < input_size; j++) {
      output_data[i * input_size + j] = 
        std::min(1.0, std::max(.0, input[i][j] / 6.0f + 0.5f));
    }
  }
  //TODO: Create tensor options with dtype float32
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  //TODO: Create a tensor from the output vector
  auto foo= torch::from_blob(output_data.data(), {int64_t(output_data.size())}, opts).clone();
  //TODO: 将得到的tensor reshape为所需的大小
  output = foo.reshape([batch_size, input_size]);
  return output;
} 
//TODO: 算子绑定为Pytorch的模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	
m.def("hsigmoid_cpu", &hsigmoid_cpu, "HSigmoid activation function (CPU)");
}       
