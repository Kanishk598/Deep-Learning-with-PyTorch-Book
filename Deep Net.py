import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = 0.1*torch.tensor(t_u).unsqueeze(1)

seq_model = nn.Sequential(
    OrderedDict([
        ("hidden_linear", nn.Linear(1, 3)),
        ("hidden_activation", nn.Tanh()),
        ("output", nn.Linear(3, 1))
    ])
)

optimizer = optim.Adam(seq_model.parameters(), lr = 1e-2)

t_range = torch.arange(20., 90.).unsqueeze(1)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(0.1*t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
#plt.plot(t_u.numpy(), seq_model(t_u).detach().numpy(), 'kx')
plt.show()

def training_loop(t_u, t_c, loss_fn, optimizer, model, n_epochs):
    for epoch in range(1, n_epochs):
        t_p = model(t_u)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ((epoch < 5) or (epoch % 500 == 0)):
            print("Loss in epoch {0} is: {1}".format(epoch, loss))

training_loop(t_u = t_u, t_c = t_c, loss_fn = nn.MSELoss(), optimizer = optimizer, model = seq_model, n_epochs = 5000)

plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(0.1*t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
#plt.plot(t_u.numpy(), seq_model(t_u).detach().numpy(), 'kx')
plt.show()