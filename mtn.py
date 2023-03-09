import torch
from torch.autograd import Variable


def tensor_hook_demo():
  grad_list = []

  def print_grad(grad):
    print("=============: grad: ", grad)
    grad_list.append(grad)
    return grad

  x = torch.rand(4, 5, requires_grad = True)
  y = x + 5

  z = torch.mean(y)

  y.register_hook(print_grad)
  z.backward()

  print("***********", grad_list[0])


def module_hook():
  """在指定网络层执行完前向传播后调用钩子函数"""
  activation = {}

  def get_activation(name):
    def hook(model, input, output):
      activation[name]=output.detach()
    return hook

  import torch
  import torchvision.models as models
  model = models.vgg16(pretrained=True)
  model.features[4].register_forward_hook(get_activation('maxpool'))
  print("============")
  input_data = torch.randn(1, 3, 224, 224)
  output = model(input_data)

  # plt image
  # maxpool = activation['maxpool']
  # plt.figure(figsize = (11, 6))
  # for i in range(maxpool.size(1)):
  #   plt.subplot(6, 11, i + 1)
  #   plt.imshow(maxpool.data.numpy()[0, i, :, :], cmap = 'gray')
  #   plt.axis('off')

  # plt.subplot_adjust(wspace=0.1, hspace=0.1)
  # plt.show()

def module_backward_hook():
  import torch
  import torch.nn as nn

  def hook_func(model, grad_input, grad_output):
    print("grad_input: ", grad_input)
    print("grad_output: ", grad_output)

  net1 = nn.Linear(4, 2)

  handle = net1.register_backward_hook(hook_func)

  x = torch.tensor([[1.0, 2, 3, 4]], requires_grad = True)
  out = net1(x)
  out.backward(torch.tensor([[2, 1]]))

if __name__ == "__main__":
  # tensor_hook_demo()
  # module_hook()
  module_backward_hook()
  print("run mtn.py successfully !!!")


