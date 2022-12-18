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
import math

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

def attention(x, wq, q_bias, wk, k_bias, wv, v_bias, dk):
    q = torch.matmul(x, wq) + q_bias
    k = torch.matmul(x, wk) + k_bias
    v = torch.matmul(x, wv) + v_bias

    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    score = torch.nn.Softmax(dim=1)(score)
    return torch.matmul(score, v)

(n, d_model, dk, transpose) = (64, 1024, 64, 1)

# Generate inputs
x = torch.randn((n, d_model)) * 3.14
wq = torch.randn((d_model, dk)) * 3.14
wk = torch.randn((d_model, dk)) * 3.14
wv = torch.randn((d_model, dk)) * 3.14
q_bias = torch.randn(dk) * 3.14 
k_bias = torch.randn(dk) * 3.14
v_bias = torch.randn(dk) * 3.14

o = 10 * torch.randn((n, dk))
o_gold = attention(x, wq, q_bias, wk, k_bias, wv, v_bias, dk)

# Sclae weight and bias of Q to avoid hardware scaling
scale = math.sqrt(dk)
wq = wq / scale
q_bias = q_bias / scale

if transpose == 1:
    x = torch.transpose(x, 0, 1)

print(".section .data,\"aw\",@progbits")
emit("n", np.array(n, dtype=np.int32))
emit("d_model", np.array(d_model, dtype=np.int32))
emit("dk", np.array(dk, dtype=np.int32))
emit("transpose", np.array(transpose, dtype=np.int32))

emit("x", x.numpy().astype(np.float32), 'NR_LANES*32')
emit("wk", wk.numpy().astype(np.float32), 'NR_LANES*32')
emit("wq", wq.numpy().astype(np.float32), 'NR_LANES*32')
emit("wv", wv.numpy().astype(np.float32), 'NR_LANES*32')
emit("q_bias", q_bias.numpy().astype(np.float32), 'NR_LANES*32')
emit("v_bias", v_bias.numpy().astype(np.float32), 'NR_LANES*32')
emit("k_bias", k_bias.numpy().astype(np.float32), 'NR_LANES*32')

emit("o_gold", o_gold.numpy().astype(np.float32), 'NR_LANES*32')
emit("o", o.numpy().astype(np.float32), 'NR_LANES*32')
