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

def attention(q, k, v):
    score = torch.matmul(q, k)
    score = torch.nn.Softmax(dim=-1)(score)
    return torch.matmul(score, v)

(n, d_model, dk, transpose) = (64, 768, 64, 1)

# Generate inputs
q = torch.randn((n, dk)) * 3.14
k = torch.randn((dk, n)) * 3.14
v = torch.randn((n, dk)) * 3.14

o = 10 * torch.randn((n, dk))
o_gold = attention(q, k, v)

if transpose == 1:
    q = torch.transpose(q, 0, 1)
    v = torch.transpose(v, 0, 1)

print(".section .data,\"aw\",@progbits")
emit("n", np.array(n, dtype=np.int32))
emit("d_model", np.array(d_model, dtype=np.int32))
emit("dk", np.array(dk, dtype=np.int32))
emit("transpose", np.array(transpose, dtype=np.int32))

emit("k", k.numpy().astype(np.float32), 'NR_LANES*32')
emit("q", q.numpy().astype(np.float32), 'NR_LANES*32')
emit("v", v.numpy().astype(np.float32), 'NR_LANES*32')

emit("o_gold", o_gold.numpy().astype(np.float32), 'NR_LANES*32')
emit("o", o.numpy().astype(np.float32), 'NR_LANES*32')
