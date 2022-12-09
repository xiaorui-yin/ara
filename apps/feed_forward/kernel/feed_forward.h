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

#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H
/*
  input:
  x: input matrix from multihead self-attention, n x d_mdoel
  w1, w2: weight matrices for liner transformation, d_model x d_ff, d_ff x
  d_model alpha, beta: layer normalization parameters

  output:
  o = ((activation(x * w1).dropout()) * w2 + x).layernorm()
*/
void feed_forward(float *x, float *o, float *w1, float *bias_1, float *w2,
                  float *bias_2, float *alpha, float *beta, int *sel,
                  const float scale, const int n, const int d_model);

#endif // !FEED_FORWARD_H
