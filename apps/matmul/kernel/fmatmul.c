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

void fmatmul(float *c, const float *a, const float *b,
             const unsigned long int M, const unsigned long int N,
             const unsigned long int P) {
  if (M <= 4) {
    fmatmul_4x4(c, a, b, M, N, P);
  } else if (M <= 8) {
    fmatmul_8x8(c, a, b, M, N, P);
  } else if (M <= 64) {
    fmatmul_16x16(c, a, b, M, N, P);
  } else if (M <= 128) {
    // Vector length is 64 elements. With an 8x8 matmul,
    // we can use LMUL=2, having a vl of 128.
    fmatmul_8x8(c, a, b, M, N, P);
  } else {
    // Vector length is 64 elements. With an 4x4 matmul,
    // we can use LMUL=4, having a vl of 256.
    fmatmul_4x4(c, a, b, M, N, P);
  }
}

// ---------------
// 4x4
// ---------------

void fmatmul_4x4(float *c, const float *a, const float *b,
                 const unsigned long int M, const unsigned long int N,
                 const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 4;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_4x4_slice_init();
      fmatmul_vec_4x4(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_vec_4x4_slice_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v12, 0");
}

void fmatmul_vec_4x4(float *c, const float *a, const float *b,
                     const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t3));
    t3 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
    t3 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
}

// ---------------
// 8x8
// ---------------

void fmatmul_8x8(float *c, const float *a, const float *b,
                 const unsigned long int M, const unsigned long int N,
                 const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 8;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_8x8_slice_init();
      fmatmul_vec_8x8(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_vec_8x8_slice_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v14, 0");
}

void fmatmul_vec_8x8(float *c, const float *a, const float *b,
                     const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v18, (%0);" ::"r"(b));
  b += P;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v20, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));
    t7 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v18, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
    t7 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
}

// ---------------
// 16x16
// ---------------

void fmatmul_16x16(float *c, const float *a, const float *b,
                   unsigned long int M, unsigned long int N,
                   unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init();
      fmatmul_vec_16x16(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_vec_16x16_slice_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v1,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v3,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v5,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v7,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v9,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v11, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v13, 0");
  asm volatile("vmv.v.i v14, 0");
  asm volatile("vmv.v.i v15, 0");
}

void fmatmul_vec_16x16(float *c, const float *a, const float *b,
                       const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a, a += N;
  t8 = *a, a += N;
  t9 = *a, a += N;
  t10 = *a, a += N;
  t11 = *a, a += N;
  t12 = *a, a += N;
  t13 = *a, a += N;
  t14 = *a, a += N;
  t15 = *a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vse32.v v1, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vse32.v v3, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vse32.v v5, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vse32.v v7, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vse32.v v9, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vse32.v v11, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vse32.v v13, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}

// =================================================
// MatMul followed by a transpose operation
// (A * B)^T
// =================================================

void fmatmul_transpose(float *c, const float *a, const float *b,
                       const unsigned long int M, const unsigned long int N,
                       const unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p * M;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m;

      fmatmul_vec_16x16_slice_init();
      fmatmul_vec_16x16_transpose(c__, a_, b_, M, N, P);
    }
  }
}

void fmatmul_vec_16x16_transpose(float *c, const float *a, const float *b,
                                 const unsigned long int M,
                                 const unsigned long int N,
                                 const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a, a += N;
  t8 = *a, a += N;
  t9 = *a, a += N;
  t10 = *a, a += N;
  t11 = *a, a += N;
  t12 = *a, a += N;
  t13 = *a, a += N;
  t14 = *a, a += N;
  t15 = *a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  int stride = 4 * M; // stride size in byte (float: 4 bytes)
  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  // asm volatile("vse32.v v0, (%0);" ::"r"(c));
  asm volatile("vsse32.v v0, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  // asm volatile("vse32.v v1, (%0);" ::"r"(c));
  asm volatile("vsse32.v v1, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  // asm volatile("vse32.v v2, (%0);" ::"r"(c));
  asm volatile("vsse32.v v2, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  // asm volatile("vse32.v v3, (%0);" ::"r"(c));
  asm volatile("vsse32.v v3, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  // asm volatile("vse32.v v4, (%0);" ::"r"(c));
  asm volatile("vsse32.v v4, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  // asm volatile("vse32.v v5, (%0);" ::"r"(c));
  asm volatile("vsse32.v v5, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  // asm volatile("vse32.v v6, (%0);" ::"r"(c));
  asm volatile("vsse32.v v6, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  // asm volatile("vse32.v v7, (%0);" ::"r"(c));
  asm volatile("vsse32.v v7, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  // asm volatile("vse32.v v8, (%0);" ::"r"(c));
  asm volatile("vsse32.v v8, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  // asm volatile("vse32.v v9, (%0);" ::"r"(c));
  asm volatile("vsse32.v v9, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  // asm volatile("vse32.v v10, (%0);" ::"r"(c));
  asm volatile("vsse32.v v10, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  // asm volatile("vse32.v v11, (%0);" ::"r"(c));
  asm volatile("vsse32.v v11, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  // asm volatile("vse32.v v12, (%0);" ::"r"(c));
  asm volatile("vsse32.v v12, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  // asm volatile("vse32.v v13, (%0);" ::"r"(c));
  asm volatile("vsse32.v v13, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  // asm volatile("vse32.v v14, (%0);" ::"r"(c));
  asm volatile("vsse32.v v14, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  // asm volatile("vse32.v v15, (%0);" ::"r"(c));
  asm volatile("vsse32.v v15, (%0), %1" ::"r"(c), "r"(stride));
}

// =================================================
// MatMul with bias
// =================================================

void fmatmul_bias(float *c, const float *a, const float *b, const float *bias,
                  unsigned long int M, unsigned long int N,
                  unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;
    const float *bias_ = bias + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init();
      fmatmul_vec_16x16_bias(c__, a_, b_, bias_, N, P);
    }
  }
}

void fmatmul_vec_16x16_bias(float *c, const float *a, const float *b,
                            const float *bias, const unsigned long int N,
                            const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a, a += N;
  t8 = *a, a += N;
  t9 = *a, a += N;
  t10 = *a, a += N;
  t11 = *a, a += N;
  t12 = *a, a += N;
  t13 = *a, a += N;
  t14 = *a, a += N;
  t15 = *a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    if (n < N) {
      // Load one row of B
      asm volatile("vle32.v v16, (%0);" ::"r"(b));
      b += P;
    } else {
      // Last MAC operation, load bias vector first
      asm volatile("vle32.v v18, (%0);" ::"r"(bias));
    }

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Last iteration: store results after add bias
  // asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vfadd.vv v0, v0, v18");
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vfadd.vv v1, v1, v18");
  asm volatile("vse32.v v1, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vfadd.vv v2, v2, v18");
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vfadd.vv v3, v3, v18");
  asm volatile("vse32.v v3, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vfadd.vv v4, v4, v18");
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vfadd.vv v5, v5, v18");
  asm volatile("vse32.v v5, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vfadd.vv v6, v6, v18");
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vfadd.vv v7, v7, v18");
  asm volatile("vse32.v v7, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vfadd.vv v8, v8, v18");
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vfadd.vv v9, v9, v18");
  asm volatile("vse32.v v9, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vfadd.vv v10, v10, v18");
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vfadd.vv v11, v11, v18");
  asm volatile("vse32.v v11, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vfadd.vv v12, v12, v18");
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vfadd.vv v13, v13, v18");
  asm volatile("vse32.v v13, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vfadd.vv v14, v14, v18");
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vfadd.vv v15, v15, v18");
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}

// =================================================
// MatMul followed by a MatAdd (A * B + D)
// =================================================

void fmatmul_add(float *c, const float *a, const float *b, const float *bias,
                 const float *d, unsigned long int M, unsigned long int N,
                 unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;
    const float *bias_ = bias + p;
    const float *d_ = d + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;
      const float *d__ = d_ + m * P;

      fmatmul_vec_16x16_slice_init();
      fmatmul_vec_16x16_add(c__, a_, b_, bias_, d__, N, P);
    }
  }
}

void fmatmul_vec_16x16_add(float *c, const float *a, const float *b,
                           const float *bias, const float *d,
                           const unsigned long int N,
                           const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a, a += N;
  t8 = *a, a += N;
  t9 = *a, a += N;
  t10 = *a, a += N;
  t11 = *a, a += N;
  t12 = *a, a += N;
  t13 = *a, a += N;
  t14 = *a, a += N;
  t15 = *a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    if (n < N) {
      // Load one row of B
      asm volatile("vle32.v v16, (%0);" ::"r"(b));
      b += P;
    } else {
      // Last MAC operation, load bias vector first
      asm volatile("vle32.v v18, (%0);" ::"r"(bias));

      // Load 13 rows of D (since only 13 vregs left)
      asm volatile("vle32.v v19, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v20, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v21, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v22, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v23, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v24, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v25, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v26, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v27, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v28, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v29, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v30, (%0);" ::"r"(d));
      d += P;
      asm volatile("vle32.v v31, (%0);" ::"r"(d));
      d += P;
    }

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Last iteration: store results after add bias
  // asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vfadd.vv v0, v0, v18");
  asm volatile("vfadd.vv v0, v0, v19");
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  // Load remaining 3 rows, same for v20, v21
  asm volatile("vle32.v v19, (%0);" ::"r"(d));
  d += P;
  c += P;
  // asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vfadd.vv v1, v1, v18");
  asm volatile("vfadd.vv v1, v1, v20");
  asm volatile("vse32.v v1, (%0);" ::"r"(c));
  asm volatile("vle32.v v20, (%0);" ::"r"(d));
  d += P;
  c += P;
  // asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vfadd.vv v2, v2, v18");
  asm volatile("vfadd.vv v2, v2, v21");
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  asm volatile("vle32.v v21, (%0);" ::"r"(d));
  d += P;
  c += P;
  // asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vfadd.vv v3, v3, v18");
  asm volatile("vfadd.vv v3, v3, v22");
  asm volatile("vse32.v v3, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vfadd.vv v4, v4, v18");
  asm volatile("vfadd.vv v4, v4, v23");
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vfadd.vv v5, v5, v18");
  asm volatile("vfadd.vv v5, v5, v24");
  asm volatile("vse32.v v5, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vfadd.vv v6, v6, v18");
  asm volatile("vfadd.vv v6, v6, v25");
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vfadd.vv v7, v7, v18");
  asm volatile("vfadd.vv v7, v7, v26");
  asm volatile("vse32.v v7, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vfadd.vv v8, v8, v18");
  asm volatile("vfadd.vv v8, v8, v27");
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vfadd.vv v9, v9, v18");
  asm volatile("vfadd.vv v9, v9, v28");
  asm volatile("vse32.v v9, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vfadd.vv v10, v10, v18");
  asm volatile("vfadd.vv v10, v10, v29");
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vfadd.vv v11, v11, v18");
  asm volatile("vfadd.vv v11, v11, v30");
  asm volatile("vse32.v v11, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vfadd.vv v12, v12, v18");
  asm volatile("vfadd.vv v12, v12, v31");
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vfadd.vv v13, v13, v18");
  asm volatile("vfadd.vv v13, v13, v19");
  asm volatile("vse32.v v13, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vfadd.vv v14, v14, v18");
  asm volatile("vfadd.vv v14, v14, v20");
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
  c += P;
  // asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vfadd.vv v15, v15, v18");
  asm volatile("vfadd.vv v15, v15, v21");
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}

// =================================================
// Special MatMul for self-attention (store result
// in the position after concatenation)
// d_model: column size of the 'big' matrix
// =================================================

void fmatmul_sv(float *c, const float *a, const float *b,
                     unsigned long int M, unsigned long int N,
                     unsigned long int P, const int d_model) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * d_model;

      fmatmul_vec_16x16_slice_init();
      fmatmul_vec_16x16_sv(c__, a_, b_, N, P, d_model);
    }
  }
}

void fmatmul_vec_16x16_sv(float *c, const float *a, const float *b,
                               const unsigned long int N,
                               const unsigned long int P, const int d_model) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a, a += N;
  t8 = *a, a += N;
  t9 = *a, a += N;
  t10 = *a, a += N;
  t11 = *a, a += N;
  t12 = *a, a += N;
  t13 = *a, a += N;
  t14 = *a, a += N;
  t15 = *a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += d_model;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v17, (%0);" ::"r"(b));
    b += d_model;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v16, (%0);" ::"r"(b));
    b += d_model;

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vse32.v v1, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vse32.v v3, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vse32.v v5, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vse32.v v7, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vse32.v v9, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vse32.v v11, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vse32.v v13, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
  c += d_model;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}

// =================================================
// A with offset=d_model
// =================================================

void fmatmul_qk(float *c, const float *a, const float *b,
                     unsigned long int M, unsigned long int N,
                     unsigned long int P, const int d_model) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * d_model;
      float *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init();
      fmatmul_vec_16x16_qk(c__, a_, b_, N, P, d_model);
    }
  }
}

void fmatmul_vec_16x16_qk(float *c, const float *a, const float *b,
                               const unsigned long int N,
                               const unsigned long int P, const int d_model) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of scalar values
  t0 = *a, a += d_model;
  t1 = *a, a += d_model;
  t2 = *a, a += d_model;
  t3 = *a, a += d_model;
  t4 = *a, a += d_model;
  t5 = *a, a += d_model;
  t6 = *a, a += d_model;
  t7 = *a, a += d_model;
  t8 = *a, a += d_model;
  t9 = *a, a += d_model;
  t10 = *a, a += d_model;
  t11 = *a, a += d_model;
  t12 = *a, a += d_model;
  t13 = *a, a += d_model;
  t14 = *a, a += d_model;
  t15 = *a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += d_model;

    // Load one row of B
    asm volatile("vle32.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += d_model;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += d_model;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += d_model;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += d_model;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += d_model;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += d_model;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += d_model;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += d_model;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += d_model;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += d_model;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += d_model;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += d_model;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += d_model;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += d_model;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += d_model;

    // Load one row of B
    asm volatile("vle32.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += d_model;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += d_model;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += d_model;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += d_model;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += d_model;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += d_model;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += d_model;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += d_model;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += d_model;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += d_model;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += d_model;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += d_model;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += d_model;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += d_model;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vse32.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vse32.v v1, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vse32.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vse32.v v3, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vse32.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vse32.v v5, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vse32.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vse32.v v7, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vse32.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vse32.v v9, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vse32.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vse32.v v11, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vse32.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vse32.v v13, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vse32.v v14, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vse32.v v15, (%0);" ::"r"(c));
}

// =================================================
// MatMul with bias, and transpose the result
// =================================================

void fmatmul_bias_transpose(float *c, const float *a, const float *b,
                            const float *bias, unsigned long int M,
                            unsigned long int N, unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p * M;
    const float *bias_ = bias + p;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m;

      fmatmul_vec_16x16_slice_init();
      fmatmul_vec_16x16_bias_transpose(c__, a_, b_, bias_, M, N, P);
    }
  }
}

void fmatmul_vec_16x16_bias_transpose(float *c, const float *a, const float *b,
                                      const float *bias,
                                      const unsigned long int M,
                                      const unsigned long int N,
                                      const unsigned long int P) {
  // Temporary variables
  float t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const float *a_ = a;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a, a += N;
  t8 = *a, a += N;
  t9 = *a, a += N;
  t10 = *a, a += N;
  t11 = *a, a += N;
  t12 = *a, a += N;
  t13 = *a, a += N;
  t14 = *a, a += N;
  t15 = *a;

  // Prefetch one row of matrix B
  asm volatile("vle32.v v16, (%0);" ::"r"(b));
  b += P;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle32.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    if (n < N) {
      // Load one row of B
      asm volatile("vle32.v v16, (%0);" ::"r"(b));
      b += P;
    } else {
      // Last MAC operation, load bias vector first
      asm volatile("vle32.v v18, (%0);" ::"r"(bias));
    }

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  int stride = 4 * M;
  // Last iteration: store results after add bias
  // asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vfadd.vv v0, v0, v18");
  asm volatile("vsse32.v v0, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vfadd.vv v1, v1, v18");
  asm volatile("vsse32.v v1, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vfadd.vv v2, v2, v18");
  asm volatile("vsse32.v v2, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vfadd.vv v3, v3, v18");
  asm volatile("vsse32.v v3, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vfadd.vv v4, v4, v18");
  asm volatile("vsse32.v v4, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vfadd.vv v5, v5, v18");
  asm volatile("vsse32.v v5, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vfadd.vv v6, v6, v18");
  asm volatile("vsse32.v v6, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vfadd.vv v7, v7, v18");
  asm volatile("vsse32.v v7, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vfadd.vv v8, v8, v18");
  asm volatile("vsse32.v v8, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vfadd.vv v9, v9, v18");
  asm volatile("vsse32.v v9, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vfadd.vv v10, v10, v18");
  asm volatile("vsse32.v v10, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vfadd.vv v11, v11, v18");
  asm volatile("vsse32.v v11, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vfadd.vv v12, v12, v18");
  asm volatile("vsse32.v v12, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vfadd.vv v13, v13, v18");
  asm volatile("vsse32.v v13, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vfadd.vv v14, v14, v18");
  asm volatile("vsse32.v v14, (%0), %1" ::"r"(c), "r"(stride));
  c++;
  // asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vfadd.vv v15, v15, v18");
  asm volatile("vsse32.v v15, (%0), %1" ::"r"(c), "r"(stride));
  c++;
}
