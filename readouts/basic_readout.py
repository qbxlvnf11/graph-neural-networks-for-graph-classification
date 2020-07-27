import torch

def readout_function(x, readout):
  if readout == 'max':
    return torch.max(x, dim=1)[0].squeeze() # max readout
  elif readout == 'avg':
    return torch.mean(x, dim=1).squeeze() # avg readout
  elif readout == 'sum':
    return torch.sum(x, dim=1).squeeze() # sum readout