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
from layernorm import LayerNorm

def emit(name, array, alignment='NR_LANES*32'):
	print(".global %s" % name)
	print(".balign " + alignment)
	print("%s:" % name)
	bs = array.tobytes()
	for i in range(0, len(bs), 4):
		s = ""
		for n in range(4):
			s += "%02x" % bs[i+3-n]
		print("    .word 0x%s" % s)

row = 64
col = 768

# Generate inputs
mat   = 3.14 * torch.randn((row, col))

alpha = 3.14 * torch.randn(col)
# beta  = 3.14 * torch.randn(col)
beta  = 314 * torch.ones(col)

# mat   = torch.FloatTensor(row, col)

# alpha = torch.FloatTensor(row, col)
# beta  = torch.FloatTensor(row, col)

# mean = mat.mean(-1, keepdim=True)
# std = 1 / torch.sqrt(mat.var(-1, keepdim=True) + 0.00001) 


kernel = LayerNorm(alpha, beta)
o_gold = kernel(mat)

print(".section .data,\"aw\",@progbits")
emit("row", np.array(row, dtype=np.int32))
emit("col", np.array(col, dtype=np.int32))
emit("mat", mat.numpy().astype(np.float32), 'NR_LANES*32')
emit("alpha", alpha.numpy().astype(np.float32), 'NR_LANES*32')
emit("beta", beta.numpy().astype(np.float32), 'NR_LANES*32')
emit("o_gold", o_gold.numpy().astype(np.float32), 'NR_LANES*32')

# emit("std", std.numpy().astype(np.float32), 'NR_LANES*32')
# emit("mean", mean.numpy().astype(np.float32), 'NR_LANES*32')
