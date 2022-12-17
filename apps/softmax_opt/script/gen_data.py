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

def softmax(mat):
    row = mat.size()[-2]
    col = mat.size()[-1]

    o = torch.Tensor(row, col)

    cur_sum = torch.zeros(row)
    cur_max = torch.ones(row) * -999999
    for i in range(col):
        data = mat[:, i]
        next_max = torch.max(cur_max, data)
        data = data - next_max
        data = torch.exp(data)
        # data = torch.ones(row) * 5

        tmp_ = cur_max - next_max
        tmp_ = torch.exp(tmp_)

        next_sum = cur_sum * tmp_
        next_sum = next_sum + data
        # next_sum = data

        cur_sum = next_sum
        cur_max = next_max
    for i in range(col):
        data = mat[:, i]
        tmp = data - cur_max
        tmp = torch.exp(tmp)

        tmp = tmp / cur_sum
        o[:, i] = tmp
        # o[:, i] = cur_sum
    return o


(row, col, transpose) = (32, 32, 1)

# Generate inputs
mat = torch.rand((row, col))

kernel = torch.nn.Softmax(dim=1)
if (transpose):
    o_gold = kernel(mat)
    o_gold = torch.transpose(o_gold, 0, 1)
    mat = torch.transpose(mat, 0, 1)
    tmp = row
    row = col
    col = tmp
else:
    o_gold = kernel(mat)

print(".section .data,\"aw\",@progbits")
emit("row", np.array(row, dtype=np.int32))
emit("col", np.array(col, dtype=np.int32))
emit("mat", mat.numpy().astype(np.float32), 'NR_LANES*32')
emit("o", mat.numpy().astype(np.float32), 'NR_LANES*32')
emit("o_gold", o_gold.numpy().astype(np.float32), 'NR_LANES*32')
emit("transpose", np.array(transpose, dtype=np.int32), 'NR_LANES*32')
