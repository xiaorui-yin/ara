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

#include "dropout.h"
#include "fmatmul.h"
#include "layernorm.h"
#include "matmul_opt.h"
#include "relu.h"
#include "riscv_vector.h"
#include <stdint.h>

void feed_forward(float *x, float *o, float *w1, float *bias_1, float *w2,
                  float *bias_2, float *alpha, float *beta, uint8_t *sel,
                  const float scale, const int n, const int d_model) {
  // d_ff: the feature size of weight matricies
  const int d_ff = d_model * 4;

  float o1[n * d_ff] __attribute__((aligned(32 * NR_LANES)));

  // =================================================
  // First Linear Transformation
  // =================================================

  // o1 = x * w1 + bias_1
  // n x d_ff = n x d_model * d_model x d_ff + n x d_ff
  fmatmul_bias(o1, x, w1, bias_1, n, d_model, d_ff);

  // =================================================
  // Activation and Dropout
  // =================================================

  relu(o1, n, d_ff);
  dropout_vec(n * d_ff, o1, scale, sel, o1);

  // =================================================
  // Second Linear Transformation and Residual Connection
  // =================================================

  // o = o1 * w2 + bias_2 + x
  fmatmul_add(o, o1, w2, bias_2, x, n, d_ff, d_model);

  // =================================================
  // Layer Normalization
  // =================================================

  layernorm(o, alpha, beta, n, d_model);
}

void feed_forward_t(float *x, float *o, float *w1, float *bias_1, float *w2,
                    float *bias_2, float *alpha, float *beta, uint8_t *sel,
                    const float scale, const int n, const int d_model) {
  // d_ff: the feature size of weight matricies
  const int d_ff = d_model * 4;

  float o1[n * d_ff] __attribute__((aligned(32 * NR_LANES)));

  // =================================================
  // First Linear Transformation
  // =================================================

  // o1 = x * w1 + bias_1
  // n x d_ff = n x d_model * d_model x d_ff + n x d_ff
  matmul_tb(o1, x, w1, bias_1, 1, n, d_model, d_ff);

  // =================================================
  // Activation and Dropout
  // =================================================

  relu(o1, d_ff, n);
  dropout_vec(n * d_ff, o1, scale, sel, o1);

  // =================================================
  // Second Linear Transformation and Residual Connection
  // =================================================

  // o = o1 * w2 + bias_2 + x
  matmul_ta(o, o1, w2, bias_2, x, 1, n, d_ff, d_model);

  // =================================================
  // Layer Normalization
  // =================================================

  layernorm_t(o, alpha, beta, d_model, n);
}
