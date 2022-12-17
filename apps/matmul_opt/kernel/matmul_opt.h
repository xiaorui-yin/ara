// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matheus Cavalcante, ETH Zurich
//         Samuel Riedel, ETH Zurich

#ifndef MATMUL_OPT_H
#define MATMUL_OPT_H

#include <stdint.h>

// C is transposed: P x M
// is_a_t: Is the matrix A transposed?
// offset: the stride size for the matrix C, by default it is m
void matmul_t(float *c, const float *a, const float *b, const int is_a_t,
              const unsigned int offset, const unsigned long int m,
              const unsigned long int n, const unsigned long int p);

// Add bias value to the result
// length of bias: M
void matmul_tb(float *c, const float *a, const float *b, const float *bias,
               const int is_a_t, const unsigned long int m,
               const unsigned long int n, const unsigned long int p);

// Add bias value and residual matrix x to the result
// length of bias: M, size of x: M x P
void matmul_ta(float *c, const float *a, const float *b, const float *bias,
               const float *x, const int is_a_t, const unsigned long int m,
               const unsigned long int n, const unsigned long int p);

void matmul_t_at(float *c, const float *a, const float *b,
                 const unsigned int offset, const unsigned long int m,
                 const unsigned long int n, const unsigned long int p);

void matmul_t_a(float *c, const float *a, const float *b,
                const unsigned int offset, const unsigned long int m,
                const unsigned long int n, const unsigned long int p);
#endif
