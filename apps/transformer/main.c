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

#include <riscv_vector.h>
#include <stdio.h>
#include "kernel/transformer.h"
#include "common/common.h"

#ifndef SPIKE
#include "printf.h"
#endif

#include "runtime.h"

extern const int n, d_model, h;
extern const float scale;
extern float x[]       __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float wq[]      __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float q_bias[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float wk[]      __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float k_bias[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float wv[]      __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float v_bias[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float wo[]      __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float o_bias[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float alpha_1[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float beta_1[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float w1[]      __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float bias_1[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float w2[]      __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float bias_2[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float beta[]    __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float alpha[]   __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int   sel[]     __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float alpha_2[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float beta_2[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float o_gold[]  __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float o[]       __attribute__((aligned(32 * NR_LANES), section(".l2")));

int main() {
  printf("\n");
  printf("========================\n");
  printf("=      Transformer     =\n");
  printf("========================\n");
  printf("\n");
  printf("\n");

#ifdef SPIKE
  // Enable V extension
  ENABLE_VEC;
#endif

#ifndef SPIKE
  start_timer();
  transformer(x, o, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias, alpha_1, beta_1, 
              w1, bias_1, w2, bias_2, alpha_2, beta_2, sel, scale, n, d_model, h);
  stop_timer();
  
  // Performance metrics
  int64_t runtime = get_timer();
   //float performance = 2.0 * CH * F * F * M * N / runtime;
   //float utilization = 100 * performance / (2.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
  //printf("The performance is %f SPFLOP/cycle (%f%% utilization).\n",
  //       performance, utilization);
#else
  transformer(x, o, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias, alpha_1, beta_1, 
              w1, bias_1, w2, bias_2, alpha_2, beta_2, sel, scale, n, d_model, h);
#endif

  printf("Verifying result\n");
  compare_matrix(o, o_gold, n, d_model);
}
