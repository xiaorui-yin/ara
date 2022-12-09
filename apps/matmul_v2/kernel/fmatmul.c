#include "fmatmul.h"
// #include "printf.h"
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
// ---------------
// 16x16
// ---------------

void fmatmul(float *c, const float *a, const float *b,
             const unsigned long int M, const unsigned long int N,
             const unsigned long int P) {
  const int REUSE_SIZE = 4;
  const int stride_a = 4 * N;
  const int stride_c = 4 * P;
  // We work on 64 elements of the matrix B at once
  const unsigned long int block_size_p = 32;
  // block_size_m <= M, REUSE_SIZE * length of b_vec * 32 < VRF capacity
  const unsigned long int block_size_m = NR_LANES * REUSE_SIZE;

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    // Set the broadcast length
    const unsigned long int p_ = MIN(P - p, block_size_p);
    int tmp1 = 0;
    int tmp2 = 0;
    asm volatile("vsetbl %0, %1, %2" ::"r"(tmp2), "r"(p_), "r"(tmp1));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size_m) {

      // Set the vector length
      const unsigned long int m_ = MIN(M - m, block_size_m);
      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(m_));

      // Find pointer to the submatrices
      const float *a_ = a + m * N;
      float *c__ = c_ + m * P;

      // First op with scalar zero
      // load vec_a
      asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_), "r"(stride_a));
      // load vec_b
      asm volatile("vle32bc.v v31, (%0)" ::"r"(b_));
      float t0 = 0; // First Operation, accumulated result is 0
      asm volatile("vfbmacc.vf v8, %0, v0" ::"f"(t0));

      // // load vec_a
      asm volatile("vlse32.v v1, (%0), %1" ::"r"(a_ + 1), "r"(stride_a));
      // load vec_b
      asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + P));
      asm volatile("vfbmacc.vv v8, v31, v1");

      for (unsigned long int n = 2; n < N; n += 2) {
        // load vec_a
        asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_ + n), "r"(stride_a));
        // load vec_b
        asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + n * P));
        asm volatile("vfbmacc.vv v8, v31, v0");

        // // load vec_a
        asm volatile("vlse32.v v1, (%0), %1" ::"r"(a_ + n + 1), "r"(stride_a));
        // load vec_b
        asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + (n + 1) * P));
        asm volatile("vfbmacc.vv v8, v31, v1");
      }

      asm volatile(
          "vsetvli zero, %0, e32, m1, ta, ma" ::"r"(block_size_p * NR_LANES));
      asm volatile("vsse32.v v8, (%0), %1" ::"r"(c__), "r"(stride_c));
      asm volatile("vsse32.v v9, (%0), %1" ::"r"(c__ + NR_LANES * P),
                   "r"(stride_c));
      asm volatile("vsse32.v v10, (%0), %1" ::"r"(c__ + 2 * NR_LANES * P),
                   "r"(stride_c));
      asm volatile("vsse32.v v11, (%0), %1" ::"r"(c__ + 3 * NR_LANES * P),
                   "r"(stride_c));
    }
  }
}
