import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from data.mnist import Net

def idx2onehot(idx, n, idx2=None, alpha = 1):

    assert torch.max(idx).item() < n
    idx=idx.cpu()
    
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
       
    try:
        #print("only go in this one")
        ans = []
        for i in range(idx.cpu().data.numpy().size):
            arr = torch.zeros(n)
            arr.scatter_(1, idx[i], alpha)
            arr.scatter_(1, idx2, 1-alpha)
            #arr[idx.cpu().data.numpy()[i][0]] = alpha
            #arr[idx2.cpu().data.numpy()[0]] = 1-alpha
            ans.append(arr)
        onehot = torch.tensor(ans)
    except Exception as e: 
        #print(e)
        onehot = torch.zeros(idx.size(0), n)
        onehot.scatter_(1, idx, 1)

    return onehot.cuda()