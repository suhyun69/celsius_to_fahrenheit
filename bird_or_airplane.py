import torch
from torchvision import datasets
from matplotlib import pyplot as plt

data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True) # <1>
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True) # <2>

print(len(cifar10))

img, label = cifar10[99]
plt.imshow(img)
plt.show()