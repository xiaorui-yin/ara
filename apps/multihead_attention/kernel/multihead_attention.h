// The APACHE License (APACHE)

// Copyright (c) 2022 Xiaorui Yin. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

/*
  input:
  x: input token sequence of output from previous layer, n x d_model
  wq, wk, wv: weight matrices, to store it contiguously in memory, the size is (d_model*h) x dk
  q/k/v_bias: bias matrices, same size as weight
  wo: d_model x d_model
  alpha, beta: layernorm parameters

  output:
  multihead_attn: n x d_model
*/
void multihead_attention(float *x, float* multihead_attn, float *wq, float *q_bias, float *wk, float *k_bias, 
                         float *wv, float *v_bias, float *wo, float *o_bias, float *alpha, float *beta,
                         int n, int d_model, int h);

#endif // !MULTIHEAD_ATTENTION_H
