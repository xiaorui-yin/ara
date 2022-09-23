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
#include "relu.h"
#include "dropout.h"

void feed_forward(float *x, float *o, float *w1, float *bias_1, float *w2, float *bias_2,
                    float *alpha, float *beta, int *sel, const float scale, const int n, const int d_model) {
  // d_ff: the feature size of weight matricies
  const int d_ff = d_model * 4;

  float o1[n * d_ff] __attribute__((aligned(32 * NR_LANES)));
  float o2[n * d_model] __attribute__((aligned(32 * NR_LANES)));
  

  // =================================================
  // First Linear Transformation
  // =================================================

  // o1 = x * w1 + bias_1
  // n x d_ff = n x d_model * d_model x d_ff + n x d_ff
  matmul_biased(x, w1, o1, bias_1, n, d_model, d_ff, 0);

  // =================================================
  // Activation and Dropout
  // =================================================

  relu(o1, n, d_ff);
  dropout(o1, sel, scale, n, d_ff);

  // =================================================
  // Second Linear Transformation
  // =================================================

  // o2 = o1 * w2 + bias_2
  // n x d_model = n x d_ff * d_ff x d_model + n x d_model
  matmul_biased(o1, w2, o2, bias_2, n, d_ff, d_model, 0);

  // =================================================
  // Residual Connection
  // =================================================

  matadd(o2, x, o, n, d_model);

  // =================================================
  // Layer Normalization
  // =================================================

  layernorm(o, alpha, beta, n, d_model);
}
