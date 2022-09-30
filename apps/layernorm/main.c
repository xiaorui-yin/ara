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
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <riscv_vector.h>
#include <stdio.h>
#include "kernel/layernorm.h"
#include "common/common.h"

#ifndef SPIKE
#include "printf.h"
#endif

#include "runtime.h"

extern const int row, col; // row & column size
extern float mat[]    __attribute__((aligned(32 * NR_LANES), section(".l2"))); // matrix data (N x M) to normalize
extern float alpha[]  __attribute__((aligned(32 * NR_LANES), section(".l2"))); // matrix data (N x M) to normalize
extern float beta[]   __attribute__((aligned(32 * NR_LANES), section(".l2"))); // matrix data (N x M) to normalize
extern float o_gold[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

int main() {
  printf("\n");
  printf("========================\n");
  printf("=        LayerNorm     =\n");
  printf("========================\n");
  printf("\n");
  printf("\n");

#ifdef SPIKE
  // Enable V extension
  ENABLE_VEC;
#endif

#ifndef SPIKE
  start_timer();
  layernorm(mat, alpha, beta, row, col);
  stop_timer();
  
  // Performance metrics
  int64_t runtime = get_timer();
  float performance = (9.0 * row * col + row * 2) / runtime;
  float performance_1 = (8.0 * row * col + row * 2) / runtime; // 1 MAC operation
  float utilization = 100 * performance_1 / (2.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %f SP-FLOP/cycle (%f%% utilization).\n",
        performance, utilization);
#else
  layernorm(mat, alpha, beta, row, col);
#endif

  printf("Verifying result\n");
  compare_matrix(mat, o_gold, row, col);
}
