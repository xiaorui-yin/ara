import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, alpha, beta, eps=1e-5):
        super(LayerNorm, self).__init__()
        # alpha and beta are trainable parameters
        #self.alpha = nn.Parameter(torch.ones(size))
        #self.beta = nn.Parameter(torch.zeros(size))
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.beta
