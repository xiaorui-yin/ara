// The APACHE License (APACHE)
//
// Copyright (c) 2022 Xiaorui Yin. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fmatmul.h"
#include "layernorm.h"
#include "matmul_opt.h"
#include "riscv_vector.h"
#include "self_attention.h"

void multihead_attention(float *x, float *o, float *wq, float *q_bias,
                         float *wk, float *k_bias, float *wv, float *v_bias,
                         float *wo, float *o_bias, float *alpha, float *beta,
                         const int n, const int d_model, const int h) {
  // column size of the Q, K, V matrices
  const int dk = d_model / h;

  float mhsa[n * d_model] __attribute__((aligned(32 * NR_LANES)));

  // =================================================
  // Calculate self-attention for each head
  // =================================================

  for (int i = 0; i < h; i++) {
    // self_attention(x, (mhsa + n * dk * i),
    // mhsa + i * dk: the position of the first element of each head
    self_attention(x, (mhsa + i * dk), (wq + d_model * dk * i),
                   (q_bias + dk * i), (wk + d_model * dk * i),
                   (k_bias + dk * i), (wv + d_model * dk * i),
                   (v_bias + dk * i), n, d_model, dk);
  }

  // =================================================
  // Linear transformation with Wo and Residual Connection
  // =================================================

  fmatmul_add(o, mhsa, wo, o_bias, x, n, d_model, d_model);

  // =================================================
  // Layer Normalization
  // =================================================

  layernorm(o, alpha, beta, n, d_model);
}

// x: d_model x n
//
void multihead_attention_t(float *x, float *o, float *wq, float *q_bias,
                           float *wk, float *k_bias, float *wv, float *v_bias,
                           float *wo, float *o_bias, float *alpha, float *beta,
                           const int n, const int d_model, const int h) {
  // column size of the Q, K, V matrices
  const int dk = d_model / h;

  float mhsa[n * d_model] __attribute__((aligned(32 * NR_LANES)));

  // =================================================
  // Calculate self-attention for each head
  // =================================================

  for (int i = 0; i < h; i++) {
    // self_attention(x, (mhsa + n * dk * i),
    // mhsa + i * dk: the position of the first element of each head
    self_attention_t(x, (mhsa + i * dk), (wq + d_model * dk * i),
                     (q_bias + dk * i), (wk + d_model * dk * i),
                     (k_bias + dk * i), (wv + d_model * dk * i),
                     (v_bias + dk * i), n, d_model, dk);
  }

  // =================================================
  // Linear transformation with Wo and Residual Connection
  // =================================================

  // mhsa is not transposed (o is transposed)
  matmul_ta(o, mhsa, wo, o_bias, x, 0, n, d_model, d_model);

  // =================================================
  // Layer Normalization
  // =================================================

  layernorm_t(o, alpha, beta, d_model, n);
}
