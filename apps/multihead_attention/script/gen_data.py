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
from multihead_attention import MultiHeadAttention

import torch.nn as nn

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

def gen_sel_mask(sel):
    # Generate the selection mask for vector data
    SEL = []
    for s in torch.reshape(sel, (-1, 8)):
        SEL_ = ''.join(reversed(['1' if x else '0' for x in s]))
        SEL.append(int(SEL_, 2))
    return SEL

(n, d_model, h, transpose) = (32, 64, 2, 0)
dk = d_model // h

# Generate inputs
x = torch.randn((n, d_model)) * 3.14
wq = torch.randn((d_model, dk * h)) * 3.14
wk = torch.randn((d_model, dk * h)) * 3.14
wv = torch.randn((d_model, dk * h)) * 3.14
q_bias = torch.randn((h * dk)) * 3.14 
k_bias = torch.randn((h * dk)) * 3.14
v_bias = torch.randn((h * dk)) * 3.14

alpha = torch.randn(d_model) * 3.14
beta = torch.randn(d_model) * 3.14

wo = torch.randn(d_model, d_model) * 3.14
o_bias = torch.randn(d_model) * 3.14

o = 10 * torch.randn((n, d_model))

kernel = MultiHeadAttention(wq, q_bias, wk, k_bias, wv, v_bias,
        wo, o_bias, alpha, beta)

(o_gold, sel, scale) = kernel(x, n, d_model, h)

wq /= math.sqrt(dk)
q_bias /= math.sqrt(dk)

if transpose == 1:
    x = torch.transpose(x, 0, 1)
    o_gold = torch.transpose(o_gold, 0, 1)
    sel = torch.transpose(sel, 0, 1)

sel = gen_sel_mask(sel)

print(".section .data,\"aw\",@progbits")
emit("n", np.array(n, dtype=np.int32))
emit("d_model", np.array(d_model, dtype=np.int32))
emit("h", np.array(h, dtype=np.int32))
emit("transpose", np.array(transpose, dtype=np.int32))
emit("scale", np.array(scale, dtype=np.float32))

emit("x", x.numpy().astype(np.float32), 'NR_LANES*32')
emit("wk", wk.numpy().astype(np.float32), 'NR_LANES*32')
emit("wq", wq.numpy().astype(np.float32), 'NR_LANES*32')
emit("wv", wv.numpy().astype(np.float32), 'NR_LANES*32')
emit("wo", wo.numpy().astype(np.float32), 'NR_LANES*32')
emit("q_bias", q_bias.numpy().astype(np.float32), 'NR_LANES*32')
emit("v_bias", v_bias.numpy().astype(np.float32), 'NR_LANES*32')
emit("k_bias", k_bias.numpy().astype(np.float32), 'NR_LANES*32')
emit("o_bias", o_bias.numpy().astype(np.float32), 'NR_LANES*32')
emit("alpha", alpha.numpy().astype(np.float32), 'NR_LANES*32')
emit("beta", beta.numpy().astype(np.float32), 'NR_LANES*32')

emit("sel", np.array(sel, dtype=np.uint8), 'NR_LANES*32')

emit("o_gold", o_gold.numpy().astype(np.float32), 'NR_LANES*32')
emit("o", o.numpy().astype(np.float32), 'NR_LANES*32')
