import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime

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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

'''
model = Net()
img, _ = cifar2[0]
print(model(img.unsqueeze(0)))
'''

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):  # 에포크 숫자는 0 대신, 1부터 n_epochs까지 루프를 돌며 부여함

        loss_train = 0.0
        for imgs, labels in train_loader:  # 데이터 로더가 만들어준 배치 안에서 데이터셋을 순회함
            outputs = model(imgs)  # 모델에 배치를 넣어줌
            loss = loss_fn(outputs, labels)  # 그리고 최소화하려는 손실값을 계산
            optimizer.zero_grad()  # 마지막에 이전 기울기 값을 지움
            loss.backward()  # 역전파 수행. 즉 신경망이 학습할 모든 파라미터에 대한 기울기를 계산함
            optimizer.step()  # 모델 업데이트
            loss_train += loss.item()  # 에포크 동안 확인한 손실값을 모두 더한다. 기울기값을 꺼내고자 .item() 값을 사용해 손실값을 파이썬 수로 변환하는 것은 중요하므로 잘 기억해두자

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))  # 배치 단위의 평균 손실값을 구하기 위해 훈련 데이터 로더의 길이로 나눈다. 이 값이 총합보다 훨씬 직관적이다

train_loader = torch.utils.data.DataLoader(cifar2
                                           , batch_size=64,
                                           shuffle=True)  # DataLoader가 cifar2 데이터셋의 예제를 배치로 묶어준다. 셔플리응로 데이터셋 예제의 순서를 섞어준다

model = Net()  # 신경망을 초기화하고
optimizer = optim.SGD(model.parameters(), lr=1e-2)  # 앞서 다뤄본 확률적 경사 하강 옵티마이저
loss_fn = nn.CrossEntropyLoss()  # 크로스 엔트로피 손실값

training_loop(  # 앞에서 정의한 훈련 루프를 호출
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
)

train_loader = torch.utils.data.DataLoader(cifar2
                                           , batch_size=64,
                                           shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar2_val
                                         , batch_size=64,
                                         shuffle=False)

def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  # 파라미터를 업데이트하지 않을 것이므로 기울기는 필요 없다
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # 가장 높은 값을 가진 인덱스를 출력한다
                total += labels.shape[0]  # 예제 수를 세어서 total을 배치 크기만큼 증가시킨다
                correct += int((predicted == labels).sum())  # 확률값이 가장 높았던 클래스와 레이블의 실측값을 비교하여 불리언 배열을 얻고, 예측값이 실측값에 맞은 경우가 배치에서 얼마나 나왔는지 세어 합친다

        print("Accuracy {}: {:.2f}".format(name , correct / total))

validate(model, train_loader, val_loader)