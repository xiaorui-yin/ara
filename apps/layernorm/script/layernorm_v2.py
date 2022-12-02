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

class LayerNorm_v2(nn.Module):
    def __init__(self, alpha, beta, eps=1e-5):
        super(LayerNorm_v2, self).__init__()
        # alpha and beta are trainable parameters
        #self.alpha = nn.Parameter(torch.ones(size))
        #self.beta = nn.Parameter(torch.zeros(size))
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, x):
        row = x.size()[-2]
        col = x.size()[-1]

        # Initialize mean and variacne with zero
        # mean = x[0, :, 0] * 0
        # var  = x[0, :, 0] * 0

        mean = x[:, 0] * 0
        var  = x[:, 0] * 0

        for i in range(col):
            # data = x[0, :, i];
            data = x[:, i];
            mean = mean + data
            var += torch.square(data - mean / (i + 1))
        var = var / (col - 1)
        mean = mean / col

        mean.unsqueeze_(1)
        var.unsqueeze_(1)

        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.beta
