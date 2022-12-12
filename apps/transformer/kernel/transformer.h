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

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

// only one sublayer
void transformer(float *x, float *o, float *wq, float *q_bias, float *wk,
                 float *k_bias, float *wv, float *v_bias, float *wo,
                 float *o_bias, float *alpha_1, float *beta_1, // mhsa weights
                 float *w1, float *bias_1, float *w2, float *bias_2,
                 float *alpha_2, float *beta_2, // feed_forward weights
                 int *sel, float scale, const int n, const int d_model,
                 const int h);

#endif // !TRANSFORMER_H
