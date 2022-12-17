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
#include "matmul_opt.h"
#include "riscv_vector.h"
#include "softmax.h"

#include <math.h>

// uncomment if running self_attention test
// #define SELF_ATTN_TEST

void self_attention(float *x, float *o, float *wq, float *q_bias, float *wk,
                    float *k_bias, float *wv, float *v_bias, int n, int d_model,
                    int dk) {
  // =================================================
  // Calculate matrices Q, K, V (x * W + bias)
  // =================================================

  float q[n * dk] __attribute__((aligned(32 * NR_LANES)));
  float k_t[dk * n] __attribute__((aligned(32 * NR_LANES)));
  float v[n * dk] __attribute__((aligned(32 * NR_LANES)));

  // Q
  fmatmul_bias(q, x, wq, q_bias, n, d_model, dk);

  // K^T
  fmatmul_bias_transpose(k_t, x, wk, k_bias, n, d_model, dk);

  // V
  fmatmul_bias(v, x, wv, v_bias, n, d_model, dk);

  // =================================================
  // Calculate A = (Q * K^T) / sqrt(dk)
  // scaling already done by weight and bias parameters
  // =================================================

  float a[n * n] __attribute__((aligned(32 * NR_LANES)));

  // float scale = 1.0 / (float)sqrt((double)dk);
  // matmul_scaled(q, k_t, a, scale, n, dk, n, 0);
  fmatmul(a, q, k_t, n, dk, n);

  // =================================================
  // A = softmax(A)
  // =================================================

  float a_[n * n] __attribute__((aligned(32 * NR_LANES)));
  softmax(a, a_, n, n);

  // =================================================
  // attention = A_ * V
  // =================================================

#ifndef SELF_ATTN_TEST
  fmatmul_concate(o, a_, v, n, n, dk, d_model);
#else
  fmatmul(o, a_, v, n, n, dk);
#endif /* !SELF_ATTN_TEST */
}

// x: d_model x n
// o: dk x n
void self_attention_t(float *x, float *o, float *wq, float *q_bias, float *wk,
                      float *k_bias, float *wv, float *v_bias, int n,
                      int d_model, int dk) {
  // =================================================
  // Calculate matrices Q, K, V (x * W + bias)
  // =================================================

  float q_t[n * dk] __attribute__((aligned(32 * NR_LANES)));
  float k_t[dk * n] __attribute__((aligned(32 * NR_LANES)));
  float v_t[n * dk] __attribute__((aligned(32 * NR_LANES)));

  // Q^T
  matmul_tb(q_t, x, wq, q_bias, 1, n, d_model, dk);

  // K^T
  matmul_tb(k_t, x, wk, k_bias, 1, n, d_model, dk);

  // V^T
  matmul_tb(v_t, x, wv, v_bias, 1, n, d_model, dk);

  // =================================================
  // Calculate A^T = (K * Q^T) / sqrt(dk)
  // scaling already done by weight and bias parameters
  // =================================================

  float a_t[n * n] __attribute__((aligned(32 * NR_LANES)));

  // float scale = 1.0 / (float)sqrt((double)dk);
  // matmul_scaled(q, k_t, a, scale, n, dk, n, 0);
  matmul_t(a_t, q_t, k_t, 1, n, n, dk, n);

  // =================================================
  // A = softmax(A)
  // =================================================

  float a_t_[n * n] __attribute__((aligned(32 * NR_LANES)));
  softmax_t(a_t, a_t_, n, n);

  // =================================================
  // attention = A_ * V
  // =================================================

#ifndef SELF_ATTN_TEST
  matmul_t(o, v_t, a_t_, 0, d_model, dk, n, n);
#else
  matmul_t(o, v_t, a_t_, 0, dk, dk, n, n);
#endif
}
