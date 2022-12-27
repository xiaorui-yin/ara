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

#include "dropout.h"
#include "fmatmul.h"
#include "layernorm.h"
#include "matmul_opt.h"
#include "riscv_vector.h"
#include "self_attention.h"
#include <stdint.h>

void multihead_attention(float *x, float *o, float *wq, float *q_bias,
                         float *wk, float *k_bias, float *wv, float *v_bias,
                         float *wo, float *o_bias, float *alpha, float *beta,
                         uint8_t *sel, const float scale, const int n,
                         const int d_model, const int h) {
  // column size of the Q, K, V matrices
  const int dk = d_model / h;

  // =================================================
  // Calculate Q, K, V
  // =================================================

  float q[n * d_model] __attribute__((aligned(32 * NR_LANES)));
  float k[n * d_model] __attribute__((aligned(32 * NR_LANES)));
  float v[n * d_model] __attribute__((aligned(32 * NR_LANES)));

  fmatmul_bias(q, x, wq, q_bias, n, d_model, dk);
  fmatmul_bias_transpose(k, x, wk, k_bias, n, d_model, dk);
  fmatmul_bias(v, x, wv, v_bias, n, d_model, dk);

  // =================================================
  // Calculate self-attention score for each head
  // =================================================

  float score[n * d_model] __attribute__((aligned(32 * NR_LANES)));

  for (int i = 0; i < h; i++) {
    // self_attention(x, (score + n * dk * i),
    // score + i * dk: the position of the first element of each head
    // FIXME
    self_attention((score + i * dk), (q + i * dk), (k + i * dk * n), (v + i * dk),
                   n, d_model, dk);
  }

  // =================================================
  // Linear transformation with Wo and Residual Connection
  // =================================================

  fmatmul_add(o, score, wo, o_bias, x, n, d_model, d_model);

  // =================================================
  // Dropout
  // =================================================

  dropout_vec(n * d_model, o, scale, sel, o);

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
                           uint8_t *sel, const float scale, const int n,
                           const int d_model, const int h) {
  // column size of the Q, K, V matrices
  const int dk = d_model / h;

  // =================================================
  // Calculate Q, K, V (transposed)
  // =================================================

  float q[n * d_model] __attribute__((aligned(32 * NR_LANES)));
  float k[n * d_model] __attribute__((aligned(32 * NR_LANES)));
  float v[n * d_model] __attribute__((aligned(32 * NR_LANES)));

  matmul_tb(q, x, wq, q_bias, 1, n, d_model, dk);
  matmul_tb(k, x, wk, k_bias, 1, n, d_model, dk);
  matmul_tb(v, x, wv, v_bias, 1, n, d_model, dk);

  // =================================================
  // Calculate self-attention score for each head
  // =================================================

  float score[n * d_model] __attribute__((aligned(32 * NR_LANES)));

  for (int i = 0; i < h; i++) {
    // self_attention(x, (score + n * dk * i),
    // score + i * dk: the position of the first element of each head
    self_attention_t((score + i * dk), (q + i * dk * n), (k + i * dk * n),
                     (v + i * dk * n), n, d_model, dk);
  }

  // =================================================
  // Linear transformation with Wo and Residual Connection
  // =================================================

  // score is not transposed (o is transposed)
  matmul_ta(o, score, wo, o_bias, x, 0, n, d_model, d_model);

  // =================================================
  // Dropout
  // =================================================

  dropout_vec(n * d_model, o, scale, sel, o);

  // =================================================
  // Layer Normalization
  // =================================================

  layernorm_t(o, alpha, beta, d_model, n);
}
