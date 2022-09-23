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


//void layernorm(float **mat, float **alpha, float **beta, int row, int col);

void relu(float **mat, int row, int col);

// input:
// x: input token embedding or output of previous sublayer (n x d_model)
// wq, wk, wv: query/key/value weight matrix (d_model x d_model/h, for each head)
// wo: weight matrix (d_model x d_model)
// n: # tokens, d_model: embedding size, h: # heads
// alpha, beta: layer normalization parameters (n x d_model)
// output:
// mhsa: multi-head self-attention
//void multi_head_self_attention(float **x, float **wq, float **wk, float **wv 
//  float **wo, float **mhsa, int n, int d_model, int h, float **alpha, float **beta);

// input:
// w1: first weight matrix (d_model x d_ff), w2: second weight matrix (d_ff x d_model)
// output:
// ffn: output of feedforward layer (n x d_model)
//void feed_forward_network(float **mhsa, float **w1, float **w2, float **ffn, 
//  int n, int d_model, int d_ff, float **alpha, float **beta);
