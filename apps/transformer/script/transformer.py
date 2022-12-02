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
from feed_forward import FeedForward
from multihead_attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self, x, n, d_model, h, wq, q_bias, wk, k_bias, wv, v_bias,
            wo, o_bias, alpha_1, beta_1, w1, bias_1, w2, bias_2, alpha_2, beta_2):
        layer1 = MultiHeadAttention(wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias, alpha_1, beta_1)
        layer2 = FeedForward(w1, bias_1, w2, bias_2, alpha_2, beta_2)
        
        out = layer1(x, n, d_model, h)
        
        return layer2(out)
