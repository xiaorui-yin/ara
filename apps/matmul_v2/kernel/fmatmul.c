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

void fmatmul(float *c, const float *a, const float *b, const unsigned long int M,
             const unsigned long int N, const unsigned long int P) {
  const int REUSE_SIZE = 1;
  const int stride_a = 4 * N;
  const int stride_c = 4 * P;
  // We work on 64 elements of the matrix B at once
  // TODO: use 64 with LMUL=m2
  const unsigned long int block_size_p = 32;
  // block_size_m <= M, REUSE_SIZE * length of b_vec * 32 < VRF capacity
  const unsigned long int block_size_m = NR_LANES * REUSE_SIZE;

  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Find pointers to the submatrices
    const float *b_ = b + p;
    float *c_ = c + p;

    // Set the broadcast length
    const unsigned long int p_ = MIN(P - p, block_size_p);
    int tmp = 0;
    asm volatile("vsetbl t0, %0, %1" ::"r"(p_), "r"(tmp));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size_m) {
      // asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(block_size_p * NR_LANES));
      // asm volatile("vmv.v.i v8, 0");
      
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
      // asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_ + 1), "r"(stride_a));
      // // load vec_b
      // asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + 1 * P));
      // asm volatile("vfbmacc.vv v8, v31, v0");
      // 
      // // load vec_a
      // asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_ + 2), "r"(stride_a));
      // // load vec_b
      // asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + 2 * P));
      // asm volatile("vfbmacc.vv v8, v31, v0");

      for (unsigned long int n = 1; n < 3; n++) {
        // load vec_a
        asm volatile("vlse32.v v0, (%0), %1" ::"r"(a_ + n), "r"(stride_a));
        // load vec_b
        asm volatile("vle32bc.v v31, (%0)" ::"r"(b_ + n * P));
        asm volatile("vfbmacc.vv v8, v31, v0");
      }

      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(block_size_p * NR_LANES));
      asm volatile("vsse32.v v8, (%0), %1" ::"r"(c__), "r"(stride_c));
    }
  }
}
