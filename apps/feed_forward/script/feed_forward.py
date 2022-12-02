# The APACHE License (APACHE)

# Copyright (c) 2022 Xiaorui Yin. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch
from dropout import Dropout
from layernorm import LayerNorm

class FeedForward(nn.Module):
    def __init__(self, w1, bias_1, w2, bias_2, alpha, beta):
        super(FeedForward, self).__init__()
        self.w1 = w1
        self.bias_1 = bias_1
        self.w2 = w2
        self.bias_2 = bias_2
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        out = torch.matmul(x, self.w1) + self.bias_1

        relu = nn.ReLU()
        out = relu(out)

        dropout = Dropout()
        (out, sel, scale) = dropout(out, 0.1)

        out = torch.matmul(out, self.w2) + self.bias_2
        out = out + x


        layernorm = LayerNorm(self.alpha, self.beta)
        out = layernorm(out)

        return (out, sel, scale)
