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

// if running self_attention test
// #define SELF_ATTN_TEST

void self_attention(float *o, float *q, float *k, float *v, int n, int d_model, int dk){

  // =================================================
  // Calculate A = (Q * K^T) / sqrt(dk)
  // scaling already done by weight and bias parameters
  // K is already transposed
  // =================================================

  float a[n * n] __attribute__((aligned(32 * NR_LANES)));

#ifndef SELF_ATTN_TEST
  fmatmul_qk(a, q, k, n, dk, n, d_model);
#else
  fmatmul(a, q, k, n, dk, n);
#endif /* !SELF_ATTN_TEST */

  // =================================================
  // A = softmax(A)
  // =================================================

  float a_[n * n] __attribute__((aligned(32 * NR_LANES)));
  softmax(a, a_, n, n);

  // =================================================
  // attention = A_ * V
  // =================================================

#ifndef SELF_ATTN_TEST
  fmatmul_sv(o, a_, v, n, n, dk, d_model);
#else
  fmatmul(o, a_, v, n, n, dk);
#endif /* !SELF_ATTN_TEST */
}

// x: d_model x n
// o: dk x n
// q, k, v are all transposed
void self_attention_t(float *o, float *q, float *k, float *v, int n,
                      int d_model, int dk) {

  // =================================================
  // Calculate A^T = (K * Q^T) / sqrt(dk)
  // scaling already done by weight and bias parameters
  // =================================================

  float a_t[n * n] __attribute__((aligned(32 * NR_LANES)));

  // float scale = 1.0 / (float)sqrt((double)dk);
  // matmul_scaled(q, k_t, a, scale, n, dk, n, 0);
  matmul_t(a_t, q, k, 1, n, n, dk, n);

  // =================================================
  // A = softmax(A)
  // =================================================

  float a_t_[n * n] __attribute__((aligned(32 * NR_LANES)));
  softmax_t(a_t, a_t_, n, n);

  // =================================================
  // attention = A_ * V
  // =================================================

#ifndef SELF_ATTN_TEST
  matmul_t(o, v, a_t_, 0, d_model, dk, n, n);
#else
  matmul_t(o, v, a_t_, 0, dk, dk, n, n);
#endif
}
