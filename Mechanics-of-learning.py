import torch
import numpy as np

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(t_u, w, b):
    return w*t_u + b

def loss(t_p, t_c):
    sq_diffs = (t_p - t_c)**2
    return sq_diffs.mean()

w = torch.ones(())
b = torch.zeros(())

lr = 1e-4

def dloss_by_dt_p(t_p, t_c):
    return 2*(t_p - t_c)/(t_p.size(0))

def dt_p_by_dw(t_u, w, b):
    return w

def dt_p_by_db(t_u, w, b):
    return 1

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dt_p = dloss_by_dt_p(t_p, t_c)
    dloss_dw = dloss_dt_p*dt_p_by_dw(t_u, w, b)
    dloss_db = dloss_dt_p*dt_p_by_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])
