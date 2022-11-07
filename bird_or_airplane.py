import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

data_path = '../data-unversioned/p1ch7/'
'''
cifar10 = datasets.CIFAR10(data_path, train=True, download=True) # <1>
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True) # <2>

# print(len(cifar10))

img, label = cifar10[99]
plt.imshow(img)
# plt.show()

to_tensor = transforms.ToTensor() # ToTensor : 넘파이 배열과 PIL 이미지를 텐서로 바꾸는 역할
img_t = to_tensor(img)

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
img_t, _ = tensor_cifar10[99]

plt.imshow(img_t.permute(1,2,0)) # C x H x W를 H x W x C로 바꿔준다
# plt.show()

imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3) # 추가 차원을 만들어 데이터셋이 반환하는 모든 텐서를 쌓아 놓자

# 이제 채널별로 평균을 쉽게 계산할 수 있다
# view(3, -1)은 세 채널은 유지하고 나머지 차원을 적절한 크기 하나로 합친다.
# 그래서 3x32x32 이미지는 3x1,024 벡터로 바뀌고 평균은 각 채널의 1,024개의 요소에 대해 계산하는 것이다
print(imgs.view(3,-1).mean(dim=1)) # tensor([0.4914, 0.4822, 0.4465])

# 표준편차 계산도 비슷하다
print(imgs.view(3,-1).std(dim=1))  # tensor([0.2470, 0.2435, 0.2616])

# 필요한 값이 있으니 이제 Normalize 변환을 초기화할 수 있다
print(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))

# ToTensor 변환에 이어 붙인다
transformed_cifar10 = datasets.CIFAR10(
    data_path,
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2616))
    ]))

img_t, _ = transformed_cifar10[99]
plt.imshow(img_t.permute(1,2,0))
plt.show()
'''

cifar10 = datasets.CIFAR10(
    data_path,
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

cifar10_val = datasets.CIFAR10(
    data_path,
    train=False,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in cifar10
          if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0, 2]]