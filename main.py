import torch
from matplotlib import pyplot as plt
import torch.optim as optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params
    # print(params)

t_un = 0.1 * t_u

# 경사 하강(SGD) 옵티마이저 사용하기
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_Rate = 1e-2
optimizer = optim.SGD([params], lr=learning_Rate)

params = training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    params=params, # 이 값이 optimizer에서 사용한 params와 동일하지 않으면, 옵티마이저는 모델이 사용하는 파라미터가 어떤 것인지 알 수 없다
    t_u=t_un,
    t_c=t_c
)
print(params)

'''
t_p = model(t_un, *params)  # 단위를 모르는 값을 정규화하여 훈련하고 있음. 인자도 언패킹하고 있음

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy()) # 알 수 없는 원본 값을 그려보고 있음
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.savefig("temp_unknown_plot.png", format="png")  # bookskip
'''
