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

  softmax(a, n, n);

  // =================================================
  // TODO dropout (optional)
  // =================================================

  // =================================================
  // attention = A_ * V
  // =================================================

#ifndef SELF_ATTN_TEST
  fmatmul_concate(o, a, v, n, n, dk, d_model);
#else
  fmatmul(o, a, v, n, n, dk);
#endif /* !SELF_ATTN_TEST */

  // Store the result in the position after concatenation

  // size_t vlmax = vsetvlmax_e32m1();
  // vfloat32m1_t partial_sum;

  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < dk;) {
  //     int vl = vlmax;
  //     if (j + vlmax > dk) vl = vsetvl_e32m1(dk - j);
  //
  //     partial_sum = vfmv_v_f_f32m1(0, vl);

  //     for (int k = 0; k < n; k++) {
  //       vfloat32m1_t b_vec = vle32_v_f32m1(&v[k * dk + j], vl);
  //       partial_sum = vfmacc_vf_f32m1(partial_sum, a[i * n + k], b_vec, vl);
  //     }

  //     #ifdef SELF_ATTN_TEST
  //       vse32_v_f32m1(&o[i * dk + j], partial_sum, vl);
  //     #else
  //       vse32_v_f32m1(&o[i * d_model + k * dk + j], partial_sum, vl);
  //     #endif /* SELF_ATTN_TEST */
  //
  //     j += vl;
  //   }
  // }
}
