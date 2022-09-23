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

from feed_forward import FeedForward

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

d_model = 768
n = 64
d_ff = 4 * d_model

# Generate inputs
x = torch.randn((n, d_model)) * 3.14
w1 = torch.randn((d_model, d_ff)) * 3.14
w2 = torch.randn((d_ff, d_model)) * 3.14

bias_1 = torch.randn(d_ff) * 3.14 
bias_2 = torch.randn(d_model) * 3.14

alpha = 10 * torch.randn(d_model)
beta = 10 * torch.randn(d_model)

o = 10 * torch.randn((n, d_model))
# o = 10 * torch.randn((n, d_ff))

kernel = FeedForward(w1, bias_1, w2, bias_2, alpha, beta)
(o_gold, sel, scale) = kernel(x)

print(".section .data,\"aw\",@progbits")
emit("n", np.array(n, dtype=np.int32))
emit("d_model", np.array(d_model, dtype=np.int32))
emit("scale", np.array(scale, dtype=np.float32))

emit("x", x.numpy().astype(np.float32), 'NR_LANES*32')
emit("w1", w1.numpy().astype(np.float32), 'NR_LANES*32')
emit("w2", w2.numpy().astype(np.float32), 'NR_LANES*32')
emit("bias_1", bias_1.numpy().astype(np.float32), 'NR_LANES*32')
emit("bias_2", bias_2.numpy().astype(np.float32), 'NR_LANES*32')

emit("alpha", alpha.numpy().astype(np.float32), 'NR_LANES*32')
emit("beta", beta.numpy().astype(np.float32), 'NR_LANES*32')

emit("sel", sel.numpy().astype(np.int32), 'NR_LANES*32')

emit("o_gold", o_gold.numpy().astype(np.float32), 'NR_LANES*32')
emit("o", o.numpy().astype(np.float32), 'NR_LANES*32')
