import torch
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1) # 1번 축에 여분의 차원을 추가한다
t_u = torch.tensor(t_u).unsqueeze(1) # 1번 축에 여분의 차원을 추가한다

# 데이터셋 나누기
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val]

# 이제 인덱스 텐서를 얻었으니, 데이터 텐서로부터 훈련셋과 검증셋을 만들어보자
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

def training_loop(n_epochs, optimizer, model, loss_fn, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):

        train_t_p = model(train_t_u)
        train_loss = loss_fn(train_t_p, train_t_c)

        with torch.no_grad(): # 콘텍스트 관리자
            val_t_p = model(val_t_u)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False # 이 블록 내에서 rquires_grad가 False로 설정을 강제한다는 상황을 점검한다

        optimizer.zero_grad()
        train_loss.backward() # 검증 데이터로는 학습하면 안 되므로 val_loss.backward()가 없다
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")

    # return params
    # print(params)

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))
optimizer = optim.SGD(seq_model.parameters(), lr=1e-3) # 안정성을 위해 학습률을 조금 떨어뜨렸다

training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(),
    train_t_u=train_t_un,
    val_t_u=val_t_un,
    train_t_c=train_t_c,
    val_t_c=val_t_c
)
# print(params)
print('output', seq_model(val_t_un))
print('answer', val_t_c)
print('hidden', seq_model.hidden_linear.weight.grad)

t_range = torch.arange(20., 90.).unsqueeze(1)

fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
plt.savefig("temp_unknown_plot.png", format="png")  # bookskip
