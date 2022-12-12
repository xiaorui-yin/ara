// The APACHE License (APACHE)

// Copyright (c) 2022 Xiaorui Yin. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "feed_forward.h"
#include "multihead_attention.h"

void transformer(float *x, float *o, float *wq, float *q_bias, float *wk,
                 float *k_bias, float *wv, float *v_bias, float *wo,
                 float *o_bias, float *alpha_1, float *beta_1, // mhsa weights
                 float *w1, float *bias_1, float *w2, float *bias_2,
                 float *alpha_2, float *beta_2, // feed_forward weights
                 int *sel, float scale, const int n, const int d_model,
                 const int h) {

  // =================================================
  // Multi-Head Self-Attention Layer
  // =================================================

  float mhsa[n * d_model] __attribute__((aligned(32 * NR_LANES)));
  multihead_attention(x, mhsa, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias,
                      alpha_1, beta_1, n, d_model, h);

  // =================================================
  // Feed Forward Layer
  // =================================================

  feed_forward(mhsa, o, w1, bias_1, w2, bias_2, alpha_2, beta_2, sel, scale, n,
               d_model);
}
