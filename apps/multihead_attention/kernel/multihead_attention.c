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

#include "riscv_vector.h"
#include "utils.h"
#include "layernorm.h"
#include "self_attention.h"

void multihead_attention(float *x, float *multihead_attn, float *wq, float *q_bias, float *wk, float *k_bias, 
                         float *wv, float *v_bias, float *wo, float *o_bias, float *alpha, float *beta,
                         const int n, const int d_model, const int h) {
  // column size of the Q, K, V matrices
  const int dk = d_model / h;

  float mhsa_1[n * d_model] __attribute__((aligned(32 * NR_LANES)));
  float mhsa_2[n * d_model] __attribute__((aligned(32 * NR_LANES)));

  // =================================================
  // Calculate self-attention for each head
  // =================================================

  for (int i = 0; i < h; i++) {
    // self_attention(x, (mhsa_1 + n * dk * i),
    self_attention(x, mhsa_1,
                   (wq + d_model * dk * i), (q_bias + dk * i), 
                   (wk + d_model * dk * i), (k_bias + dk * i),
                   (wv + d_model * dk * i), (v_bias + dk * i),
                   n, d_model, dk, i);
  }

  // // =================================================
  // // Concatenate all the heads
  // // =================================================
  // 

  // for (int k = 0; k < h; k++) {
  //   for (int i = 0; i < n; i++) {
  //     for (int j = 0; j < dk;) {
  //       int vl = vsetvlmax_e32m1();
  //       if (j + vl > dk) vl = vsetvl_e32m1(dk - j);

  //       vfloat32m1_t vec = vle32_v_f32m1(&mhsa_1[k * n * dk + i * dk + j], vl);
  //       vse32_v_f32m1(&mhsa_2[d_model * i + k * dk + j], vec, vl);

  //       j += vl;
  //     }
  //   }
  // }

  // =================================================
  // Linear transformation with Wo
  // =================================================

  matmul_biased(mhsa_1, wo, mhsa_2, o_bias, n, d_model, d_model, 0);

  // =================================================
  // Residual Connection
  // =================================================

  matadd(mhsa_2, x, multihead_attn, n, d_model);

  // =================================================
  // Layer Normalization
  // =================================================

  layernorm(multihead_attn, alpha, beta, n, d_model);
}
