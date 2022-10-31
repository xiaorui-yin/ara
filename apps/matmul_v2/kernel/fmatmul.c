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

#include "fmatmul.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
// ---------------
// 16x16
// ---------------

// // Pseudo Code
// for (j in D3, j+=block_size) {
//   for (i in D1, i+=vl) {
//     vec_c = spvec_init(0, block_size); // Initialize scratch-pad register
//     for (k in D2, k++) {
//       vec_a = stride_load(a + i * D1 + k, D1, vl);
//       vec_b = broad_load(b + j * D3 + k, block_size);
//       vfmacc.vvf(vec_c, vec_a, vec_b, block_size);
//     }
//     // c[i:i+vl][j:j+block_size]
//     store_spreg(vec_c, c + i * D1 + D3, D1, block_size, vl);
//   }
// }

void fmatmul(float *c, const float *a, const float *b, unsigned long int M,
             unsigned long int N, unsigned long int P) {
  const int REUSE_SIZE = 1;
  // We work on 64 elements of the matrix B at once
  unsigned long int block_size_p = 32;
  const unsigned long int block_size_m = NR_LANES * REUSE_SIZE;
  // block_size_m <= M, REUSE_SIZE * length of b_vec * 32 < VRF capacity

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size_m) {
      // Set the vector length
      const unsigned long int m_ = MIN(M - m, block_size_m);

      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(m_));

      // broadcast length = p_
      int tmp = 0;
      asm volatile("vsetbl t0, %0, %1" ::"r"(p_), "r"(tmp));

      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      int stride = 4 * N;

      // First op with scalar zero
      // load vec_a
      asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_), "r"(stride));
      // load vec_b
      asm volatile("vle32bc.v v31, (%0)" ::"r"(b_));
      float t0 = 0; // First Operation, accumulated result is 0
      asm volatile("vfbmacc.vf v8, %0, v0" ::"f"(t0));

      for (unsigned long int n = 1; n < N; n++) {
        // load vec_a
        asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_ + n), "r"(stride));
        // load vec_b
        asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + n * N));
        asm volatile("vfbmacc.vv v8, v31, v0");
      }

      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(block_size_p));
      asm volatile("vsse32.v v8, (%0), %1" ::"r"(c__), "r"(stride));
    }
  }
}
