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
#include "kernel/self_attention.h"
#include <riscv_vector.h>
#include <stdio.h>

#ifndef SPIKE
#include "printf.h"
#endif

#include "runtime.h"

#define CHECK

extern const int n, d_model, dk;
extern const int transpose;
extern float q[] __attribute__((aligned(32 * NR_LANES)));
extern float k[] __attribute__((aligned(32 * NR_LANES)));
extern float v[] __attribute__((aligned(32 * NR_LANES)));
extern float o_gold[] __attribute__((aligned(32 * NR_LANES)));
extern float o[] __attribute__((aligned(32 * NR_LANES)));

int main() {
  printf("\n");
  printf("========================\n");
  printf("=     Self-Attention   =\n");
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
    self_attention(o, q, k, v, n, d_model, dk);
  else
    self_attention_t(o, q, k, v, n, d_model, dk);
  stop_timer();

  // Performance metrics
  int64_t runtime = get_timer();
  float softmax_ops = n * n * 32;
  float softmax_ops_ = n * n * 25;
  float performance = (2.0 * n * dk * n * 2 + softmax_ops) / (float)runtime;
  float performance_ = (1.0 * n * dk * n * 2 + softmax_ops_) / (float)runtime;
  float utilization = 100.0 * performance_ / (2.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %f SPFLOP/cycle (%f%% utilization).\n",
         performance, utilization);
#else
  if (transpose == 0)
    self_attention(o, q, k, v, n, d_model, dk);
  else
    self_attention_t(o, q, k, v, n, d_model, dk);
#endif

#ifdef CHECK
  printf("Verifying result\n");
  compare_matrix(o, o_gold, n, dk);
#endif
}
