#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# arg1: image size, arg2: filter size

import numpy as np
import sys
import torch
import torch.nn as nn
from layernorm_v2 import LayerNorm_v2

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
        # var = x.var(-1, keepdim=True, unbiased=False)
        var = x.var(-1, keepdim=True)
        var = (var * 3) / 4
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.beta


row = 1
col = 4

# Generate inputs
mat   = 31.4 * torch.randn((row, col))

# mat = torch.tensor([[1.5,.0,.0,.0]])

print("matrix:")
print(mat)
# alpha = 3.14 * torch.randn((row, col))
alpha = torch.ones(col)
# beta  = 3.14 * torch.randn((row, col))
beta  = torch.zeros(col)

kernel = LayerNorm(alpha, beta)
# kernel_2 = LayerNorm_v2(alpha, beta, eps=1e-12)
o_gold = kernel(mat)
# o = kernel_2(mat)

print("vanilla layernorm")
print(o_gold)

# print("new layernorm")
# print(o)

kernel_3 = nn.LayerNorm(col, elementwise_affine=False)
print("torch layernorm")
print(kernel_3(mat))
