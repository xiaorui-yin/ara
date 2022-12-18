#include "matmul_opt.h"
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void matmul_t(float *c, const float *a, const float *b, const int is_a_t,
              const unsigned int offset, const unsigned long int M,
              const unsigned long int N, const unsigned long int P) {

  if (is_a_t == 1)
    matmul_t_at(c, a, b, offset, M, N, P);
  else
    matmul_t_a(c, a, b, offset, M, N, P);
}

void matmul_tb(float *c, const float *a, const float *b, const float *bias,
               const int is_a_t, const unsigned long int m,
               const unsigned long int n, const unsigned long int p) {
#ifdef VCD_DUMP
  event_trigger = +1
#endif
                  matmul_t(c, a, b, is_a_t, m, m, n, p);

  unsigned int vl;
  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(p));

  // Process 2 rows at a time
  for (unsigned int i = 0; i < p; i += 2) {
    for (unsigned int j = 0; j < m;) {
      vl = MIN(m - j, vl);
      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vl));

      asm volatile("vle32.v v0, (%0)" ::"r"(c + i * m + j));
      asm volatile("vle32.v v1, (%0)" ::"r"(c + (i + 1) * m + j));
      asm volatile("vfadd.vf v2, v0, %[A]" ::[A] "f"(bias[i]));
      asm volatile("vse32.v v2, (%0);" ::"r"(c + i * m + j));
      asm volatile("vfadd.vf v3, v1, %[A]" ::[A] "f"(bias[i + 1]));
      asm volatile("vse32.v v3, (%0);" ::"r"(c + (i + 1) * m + j));

      j += vl;
    }
  }

#ifdef VCD_DUMP
  event_trigger = -1
#endif
}

void matmul_ta(float *c, const float *a, const float *b, const float *bias,
               const float *x, const int is_a_t, const unsigned long int m,
               const unsigned long int n, const unsigned long int p) {

#ifdef VCD_DUMP
  event_trigger = +1
#endif

                  matmul_t(c, a, b, is_a_t, m, m, n, p);

  unsigned int vl;
  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(p));

  // Process 2 rows at a time
  for (unsigned int i = 0; i < p; i += 2) {
    for (unsigned int j = 0; j < m;) {
      vl = MIN(m - j, vl);
      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vl));

      asm volatile("vle32.v v0, (%0)" ::"r"(c + i * m + j));
      asm volatile("vle32.v v1, (%0)" ::"r"(c + (i + 1) * m + j));

      asm volatile("vle32.v v4, (%0)" ::"r"(x + i * m + j));
      asm volatile("vfadd.vf v2, v0, %[A]" ::[A] "f"(bias[i]));
      asm volatile("vfadd.vv v2, v2, v4");
      asm volatile("vse32.v v2, (%0);" ::"r"(c + i * m + j));

      asm volatile("vle32.v v5, (%0)" ::"r"(x + (i + 1) * m + j));
      asm volatile("vfadd.vf v3, v1, %[A]" ::[A] "f"(bias[i + 1]));
      asm volatile("vfadd.vv v3, v3, v5");
      asm volatile("vse32.v v3, (%0);" ::"r"(c + (i + 1) * m + j));

      j += vl;
    }
  }

#ifdef VCD_DUMP
  event_trigger = -1
#endif
}

void matmul_t_at(float *c, const float *a, const float *b,
                 const unsigned int offset, const unsigned long int M,
                 const unsigned long int N, const unsigned long int P) {

#ifdef VCD_DUMP
  event_trigger = +1
#endif

                  const int REUSE_SIZE = 4;
  const int stride_c = 4 * offset;
  const int stride_a = 4 * N;
  // We work on 64 elements of the matrix B at once
  const unsigned long int block_size_p = 32;
  // block_size_m <= M, REUSE_SIZE * length of b_vec * 32 < VRF capacity
  const unsigned long int block_size_m = NR_LANES * REUSE_SIZE;

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p * offset;

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
      const float *a_ = a + m;
      float *c__ = c_ + m;

      // First op with scalar zero
      // load vec_b
      asm volatile("vle32bc.v v31, (%0)" ::"r"(b_));

      // load vec_a
      asm volatile("vle32.v v0, (%0)" ::"r"(a_));
      float t0 = 0; // First Operation, accumulated result is 0
      asm volatile("vfbmacc.vf v8, %0, v0" ::"f"(t0));

      // load vec_b
      asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + P));
      // load vec_a
      asm volatile("vle32.v v1, (%0)" ::"r"(a_ + M));
      asm volatile("vfbmacc.vv v8, v31, v1");

      for (unsigned long int n = 2; n < N; n += 2) {
        // load vec_b
        asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + n * P));
        // load vec_a
        asm volatile("vle32.v v0, (%0)" ::"r"(a_ + n * M));
        asm volatile("vfbmacc.vv v8, v31, v0");

        // load vec_b
        asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + (n + 1) * P));
        // load vec_a
        asm volatile("vle32.v v1, (%0)" ::"r"(a_ + (n + 1) * M));
        asm volatile("vfbmacc.vv v8, v31, v1");
      }

      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_ * NR_LANES));
      asm volatile("vsse32.v v8, (%0), %1" ::"r"(c__), "r"(stride_c));
      asm volatile("vsse32.v v9, (%0), %1" ::"r"(c__ + NR_LANES),
                   "r"(stride_c));
      asm volatile("vsse32.v v10, (%0), %1" ::"r"(c__ + 2 * NR_LANES),
                   "r"(stride_c));
      asm volatile("vsse32.v v11, (%0), %1" ::"r"(c__ + 3 * NR_LANES),
                   "r"(stride_c));
    }
  }

  // Reset broadcast length
  int tmp1 = 0;
  int tmp2 = 0;
  asm volatile("vsetbl %0, %1, %2" ::"r"(tmp2), "r"(0), "r"(tmp1));

#ifdef VCD_DUMP
  event_trigger = -1
#endif
}

void matmul_t_a(float *c, const float *a, const float *b,
                const unsigned int offset, const unsigned long int M,
                const unsigned long int N, const unsigned long int P) {

#ifdef VCD_DUMP
  event_trigger = +1
#endif

                  const int REUSE_SIZE = 4;
  const int stride_c = 4 * offset;
  const int stride_a = 4 * N;
  // We work on 64 elements of the matrix B at once
  const unsigned long int block_size_p = 32;
  // block_size_m <= M, REUSE_SIZE * length of b_vec * 32 < VRF capacity
  const unsigned long int block_size_m = NR_LANES * REUSE_SIZE;

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p * offset;

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
      float *c__ = c_ + m;

      // First op with scalar zero
      // load vec_b
      asm volatile("vle32bc.v v31, (%0)" ::"r"(b_));

      // load vec_a
      asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_), "r"(stride_a));
      float t0 = 0; // First Operation, accumulated result is 0
      asm volatile("vfbmacc.vf v8, %0, v0" ::"f"(t0));

      // load vec_b
      asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + P));
      // load vec_a
      asm volatile("vlse32.v v1, (%0), %1" ::"r"(a_ + 1), "r"(stride_a));
      asm volatile("vfbmacc.vv v8, v31, v1");

      for (unsigned long int n = 2; n < N; n += 2) {
        // load vec_b
        asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + n * P));
        // load vec_a
        asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_ + n), "r"(stride_a));
        asm volatile("vfbmacc.vv v8, v31, v0");

        // load vec_b
        asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + (n + 1) * P));
        // load vec_a
        asm volatile("vlse32.v v1, (%0), %1" ::"r"(a_ + n + 1), "r"(stride_a));
        asm volatile("vfbmacc.vv v8, v31, v1");
      }

      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(p_ * NR_LANES));
      asm volatile("vsse32.v v8, (%0), %1" ::"r"(c__), "r"(stride_c));
      asm volatile("vsse32.v v9, (%0), %1" ::"r"(c__ + NR_LANES),
                   "r"(stride_c));
      asm volatile("vsse32.v v10, (%0), %1" ::"r"(c__ + 2 * NR_LANES),
                   "r"(stride_c));
      asm volatile("vsse32.v v11, (%0), %1" ::"r"(c__ + 3 * NR_LANES),
                   "r"(stride_c));
    }
  }

  // Reset broadcast length
  int tmp1 = 0;
  int tmp2 = 0;
  asm volatile("vsetbl %0, %1, %2" ::"r"(tmp2), "r"(0), "r"(tmp1));

#ifdef VCD_DUMP
  event_trigger = -1
#endif
}
