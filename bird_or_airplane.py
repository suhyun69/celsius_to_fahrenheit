import torch
from torchvision import datasets
from matplotlib import pyplot as plt
from torchvision import transforms

data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True) # <1>
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True) # <2>

print(len(cifar10))

img, label = cifar10[99]
plt.imshow(img)
# plt.show()

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
imt_t, _ = tensor_cifar10[99]

plt.imshow(img_t.permute(1,2,0)) # C x H x W를 H x W x C로 바꿔준다
plt.show()
