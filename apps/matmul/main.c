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
#include "kernel/fmatmul.h"
#include <riscv_vector.h>
#include <stdio.h>
#ifndef SPIKE
#include "printf.h"
#endif

#include "runtime.h"

extern const int dim1, dim2, dim3;
extern const int func;
extern float mat_a[] __attribute__((aligned(32 * NR_LANES)));
extern float mat_b[] __attribute__((aligned(32 * NR_LANES)));
extern float mat_c[] __attribute__((aligned(32 * NR_LANES)));
extern float bias[] __attribute__((aligned(32 * NR_LANES)));
extern float o_gold[] __attribute__((aligned(32 * NR_LANES)));
extern float o[] __attribute__((aligned(32 * NR_LANES)));

int main() {
  printf("\n");
  printf("========================\n");
  printf("=        MatMul        =\n");
  printf("========================\n");
  printf("\n");
  printf("\n");

#ifdef SPIKE
  // Enable V extension
  ENABLE_VEC;
#endif

#ifndef SPIKE
  start_timer();
  switch (func) {
  case 0:
    fmatmul(o, mat_a, mat_b, dim1, dim2, dim3);
    break;
  case 1:
    fmatmul_bias(o, mat_a, mat_b, bias, dim1, dim2, dim3);
    break;
  case 2:
    fmatmul_transpose(o, mat_a, mat_b, dim1, dim2, dim3);
    break;
  case 3:
    fmatmul_add(o, mat_a, mat_b, bias, mat_c, dim1, dim2, dim3);
  }
  stop_timer();

  // Performance metrics
  int64_t runtime = get_timer();
  float performance, performance_;

  switch (func) {
  case 0:
    performance = (float)(2 * dim1 * dim2 * dim3) / runtime;
    performance_ = (float)(dim1 * dim2 * dim3) / runtime;
    break;
  case 1:
    performance = (float)(2 * dim1 * dim2 * dim3 + dim1 * dim3) / runtime;
    performance_ = (float)(dim1 * dim2 * dim3 + dim1 * dim3) / runtime;
    break;
  case 3:
    performance = (float)(2 * dim1 * dim2 * dim3 + 2 * dim1 * dim3) / runtime;
    performance_ = (float)(dim1 * dim2 * dim3 + 2 * dim1 * dim3) / runtime;
    break;
  }

  float utilization = 100.0 * performance_ / (2.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %f SPFLOP/cycle (%f%% utilization).\n",
         performance, utilization);
#else
  switch (func) {
  case 0:
    fmatmul(o, mat_a, mat_b, dim1, dim2, dim3);
    break;
  case 1:
    fmatmul_bias(o, mat_a, mat_b, bias, dim1, dim2, dim3);
    break;
  case 2:
    fmatmul_transpose(o, mat_a, mat_b, dim1, dim2, dim3);
    break;
  case 3:
    fmatmul_add(o, mat_a, mat_b, bias, mat_c, dim1, dim2, dim3);
  }
#endif

  printf("Verifying result\n");
  compare_matrix(o, o_gold, dim1, dim3);
}
