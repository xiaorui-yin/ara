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
#include "kernel/softmax.h"
#include "common/common.h"

#ifndef SPIKE
#include "printf.h"
#endif

#include "runtime.h"

extern const int row, col;
extern float mat[]    __attribute__((aligned(32 * NR_LANES)));
extern float o_gold[] __attribute__((aligned(32 * NR_LANES)));

int main() {
  printf("\n");
  printf("========================\n");
  printf("=        Softmax       =\n");
  printf("========================\n");
  printf("\n");
  printf("\n");

#ifdef SPIKE
  // Enable V extension
  ENABLE_VEC;
#endif

#ifndef SPIKE
  start_timer();
  // softmax(mat, row, col);
  softmax_transposed(mat, row, col);
  stop_timer();
  
  // Performance metrics
  int64_t runtime = get_timer();
  float performance = (row * col * (3*28 + 7)) / (float)runtime;
  float performance_1 = (row * col * (3*21 + 7)) / (float)runtime;
  float utilization = 100.0 * performance_1 / (2.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %f SPFLOP/cycle (%f%% utilization).\n",
        performance, utilization);
#else
  softmax_transposed(mat, row, col);
  // softmax(mat, row, col);
#endif
  printf("Verifying result\n");
  compare_matrix(mat, o_gold, row, col);
}
