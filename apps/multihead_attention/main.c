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

#include "common/common.h"
#include "kernel/multihead_attention.h"
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>

#ifndef SPIKE
#include "printf.h"
#endif

#include "runtime.h"

extern const int n, d_model, h;
extern const int transpose;
extern const float scale;
extern float x[] __attribute__((aligned(32 * NR_LANES)));
extern float wq[] __attribute__((aligned(32 * NR_LANES)));
extern float q_bias[] __attribute__((aligned(32 * NR_LANES)));
extern float wk[] __attribute__((aligned(32 * NR_LANES)));
extern float k_bias[] __attribute__((aligned(32 * NR_LANES)));
extern float wv[] __attribute__((aligned(32 * NR_LANES)));
extern float v_bias[] __attribute__((aligned(32 * NR_LANES)));
extern float wo[] __attribute__((aligned(32 * NR_LANES)));
extern float o_bias[] __attribute__((aligned(32 * NR_LANES)));
extern float alpha[] __attribute__((aligned(32 * NR_LANES)));
extern float beta[] __attribute__((aligned(32 * NR_LANES)));
extern uint8_t sel[] __attribute__((aligned(32 * NR_LANES)));
extern float o_gold[] __attribute__((aligned(32 * NR_LANES)));
extern float o[] __attribute__((aligned(32 * NR_LANES)));

int main() {
  printf("\n");
  printf("========================\n");
  printf("=  Multihead-Attention =\n");
  printf("========================\n");
  printf("\n");
  printf("\n");

#ifdef SPIKE
  // Enable V extension
  ENABLE_VEC;
#endif

#ifndef SPIKE
  start_timer();
  if (transpose == 0)
    multihead_attention(x, o, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias,
                        alpha, beta, sel, scale, n, d_model, h);
  else
    multihead_attention_t(x, o, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias,
                          alpha, beta, sel, scale, n, d_model, h);

  stop_timer();

  // Performance metrics
  int64_t runtime = get_timer();

  float dk = d_model / h;

  float softmax_ops = n * n * 32;
  float softmax_ops_ = n * n * 25;
  float self_attention_ops = 2.0 * n * dk * n * 2 + softmax_ops;
  float self_attention_ops_ = 1.0 * n * dk * n * 2 + softmax_ops_;

  float layernorm_ops = (7.0 * n * d_model + n * 4);
  float layernorm_ops_ = layernorm_ops;

  float other_ops =
      2.0 * n * d_model * d_model * 4 + n * d_model * 5 + n * d_model;
  float other_ops_ =
      1.0 * n * d_model * d_model * 4 + n * d_model * 5 + n * d_model;

  float performance =
      (float)(self_attention_ops * h + layernorm_ops + other_ops) / runtime;
  float performance_ =
      (float)(self_attention_ops_ * h + layernorm_ops_ + other_ops_) / runtime;
  float utilization = (float)100 * performance_ / (2.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %f SPFLOP/cycle (%f%% utilization).\n",
         performance, utilization);
#else
  if (transpose == 0)
    multihead_attention(x, o, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias,
                        alpha, beta, sel, scale, n, d_model, h);
  else
    multihead_attention_t(x, o, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias,
                          alpha, beta, sel, scale, n, d_model, h);
#endif

  printf("Verifying result\n");
  compare_matrix(o, o_gold, n, d_model);
}
