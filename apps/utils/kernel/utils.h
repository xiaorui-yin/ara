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

#ifndef UTILS_H
#define UTILS_H

void matadd(float *mat_a, float *mat_b, float *o, int row, int col);

// mat_a: dim1 x dim2
// mat_b: dim2 x dim3
// o    : dim1 x dim3 (transposed dim3 x dim1)
void matmul(float *mat_a, float *mat_b, float *o, int dim1, int dim2, int dim3, int transposed);

// add bias matrix to the matmul result
void matmul_biased(float *mat_a, float *mat_b, float *o, float *bias, int dim1, int dim2, int dim3, int transposed);

// scale the matmul result
void matmul_scaled(float *mat_a, float *mat_b, float *o, float scale, int dim1, int dim2, int dim3, int transposed);

void transpose(float *mat, float *o, int row, int col);

#endif
