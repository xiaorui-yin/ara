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

import torch
import torch.nn as nn
import math

from layernorm import LayerNorm

def attention(x, wq, q_bias, wk, k_bias, wv, v_bias, dk):
    q = torch.matmul(x, wq) + q_bias
    k = torch.matmul(x, wk) + k_bias
    v = torch.matmul(x, wv) + v_bias

    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    score = torch.nn.Softmax(dim=1)(score)
    return torch.matmul(score, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, wq, q_bias, wk, k_bias, wv, v_bias,
            wo, o_bias, alpha, beta):
        super(MultiHeadAttention, self).__init__()
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_bias = q_bias
        self.k_bias = k_bias
        self.v_bias = v_bias
        self.o_bias = o_bias
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, n, d_model, h):
        dk = d_model // h
        score = torch.FloatTensor(n, d_model)

        for i in range(h):
            wq_i = self.wq[i*d_model:(i*d_model+d_model), :]
            wk_i = self.wk[i*d_model:(i*d_model+d_model), :]
            wv_i = self.wv[i*d_model:(i*d_model+d_model), :]
            q_bias_i = self.q_bias[i]
            k_bias_i = self.k_bias[i]
            v_bias_i = self.v_bias[i]

            score[:, i*dk:(i*dk + dk)] = attention(
                    x, wq_i, q_bias_i, wk_i, k_bias_i,
                    wv_i, v_bias_i, dk)

        score = torch.matmul(score, self.wo) + self.o_bias
        score = score + x
        layernorm = LayerNorm(self.alpha, self.beta)
        score = layernorm(score)
        return score
