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

from typing import Match
import numpy as np
import sys
import torch

torch.set_default_dtype(torch.float32)

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

# func: bias(1) biasAdd(2)
(dim1, dim2, dim3, func) = (32, 64, 64, 2)
is_a_t = 0

# Generate inputs
mat_a = 10 * torch.rand((dim1, dim2))
mat_b = 10 * torch.rand((dim2, dim3))
mat_c = 10 * torch.rand((dim1, dim3))
bias = torch.randn(dim3) * 3.14
o = torch.rand((dim1, dim3))

o_gold = torch.matmul(mat_a, mat_b)


if func == 1:
    o_gold = o_gold + bias
elif func == 2:
    o_gold = o_gold + bias + mat_c

o_gold = torch.transpose(o_gold, 0, 1)
mat_c = torch.transpose(mat_c, 0, 1)
if is_a_t:
    mat_a = torch.transpose(mat_a, 0, 1)

print(".section .data,\"aw\",@progbits")
emit("dim1", np.array(dim1, dtype=np.int32))
emit("dim2", np.array(dim2, dtype=np.int32))
emit("dim3", np.array(dim3, dtype=np.int32))

emit("func", np.array(func, dtype=np.int32))

emit("mat_a", mat_a.numpy().astype(np.float32), 'NR_LANES*32')
emit("mat_b", mat_b.numpy().astype(np.float32), 'NR_LANES*32')
emit("mat_c", mat_c.numpy().astype(np.float32), 'NR_LANES*32')
emit("bias", bias.numpy().astype(np.float32), 'NR_LANES*32')
emit("o_gold", o_gold.numpy().astype(np.float32), 'NR_LANES*32')
emit("o", o.numpy().astype(np.float32), 'NR_LANES*32')
