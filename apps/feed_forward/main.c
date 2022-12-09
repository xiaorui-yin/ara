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
#include "kernel/feed_forward.h"
#include <riscv_vector.h>
#include <stdio.h>

#ifndef SPIKE
#include "printf.h"
#endif

#include "runtime.h"

extern const int n, d_model;
extern const float scale;
extern float x[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float w1[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float bias_1[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float w2[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float bias_2[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float beta[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float alpha[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int sel[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float o_gold[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float o[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

int main() {
  printf("\n");
  printf("========================\n");
  printf("=   Feed Forward Layer =\n");
  printf("========================\n");
  printf("\n");
  printf("\n");

#ifdef SPIKE
  // Enable V extension
  ENABLE_VEC;
#endif

#ifndef SPIKE
  start_timer();
  feed_forward(x, o, w1, bias_1, w2, bias_2, alpha, beta, sel, scale, n,
               d_model);
  stop_timer();

  // Performance metrics
  int64_t runtime = get_timer();
  float d_ff = 4.0 * d_model;
  float layernorm_ops = (9.0 * n * d_model + n * 2);
  float layernorm_ops_ = (8.0 * n * d_model + n * 2); // 1 MAC operation
  float performance = (layernorm_ops + 4 * n * d_ff * d_model + n * d_ff +
                       n * d_model + 2 * n * d_ff) /
                      runtime;
  float performance_ = (layernorm_ops_ + 2 * n * d_ff * d_model + n * d_ff +
                        n * d_model + 2 * n * d_ff) /
                       runtime;
  float utilization = 100.0 * performance_ / (2.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %f SPFLOP/cycle (%f%% utilization).\n",
         performance, utilization);
#else
  feed_forward(x, o, w1, bias_1, w2, bias_2, alpha, beta, sel, scale, n,
               d_model);
#endif

  printf("Verifying result\n");
  compare_matrix(o, o_gold, n, d_model);
  // compare_matrix(o, o_gold, n, d_model*4);
}
