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

#include "riscv_vector.h"

void matadd(float *mat_a, float *mat_b, float *o, int row, int col) {
  size_t vlmax = vsetvlmax_e32m1();
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col;) {
      int vl = vlmax;
      if (j + vlmax > col) vl = vsetvl_e32m1(col - i);

      vfloat32m1_t a_vec = vle32_v_f32m1(&mat_a[i * col + j], vl);
      vfloat32m1_t b_vec = vle32_v_f32m1(&mat_b[i * col + j], vl);
      
      vfloat32m1_t o_vec = vfadd_vv_f32m1(a_vec, b_vec, vl);
      
      vse32_v_f32m1(&o[i * col + j], o_vec, vl);

      j += vl;
    }
  }
}

// mat_a: dim1 x dim2
// mat_b: dim2 x dim3
// o    : dim1 x dim3 (transposed dim3 x dim1)
void matmul(float *mat_a, float *mat_b, float *o, int dim1, int dim2, int dim3, int transposed) {
  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t partial_sum;

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim3;) {
      int vl = vlmax;
      if (j + vlmax > dim3) vl = vsetvl_e32m1(dim3 - j);
      
      partial_sum = vfmv_v_f_f32m1(0, vl);

      for (int k = 0; k < dim2; k++) {
        vfloat32m1_t b_vec = vle32_v_f32m1(&mat_b[k * dim3 + j], vl);
        partial_sum = vfmacc_vf_f32m1(partial_sum, mat_a[i * dim2 + k], b_vec, vl);
      }

      if (transposed == 0) vse32_v_f32m1(&o[i * dim3 + j], partial_sum, vl);
      else vsse32_v_f32m1(&o[j * dim1 + i], dim1 * 4, partial_sum, vl);
      j += vl;
    }
  }
}

// add bias matrix to the matmul result
void matmul_biased(float *mat_a, float *mat_b, float *o, float *bias, int dim1, int dim2, int dim3, int transposed) {
  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t partial_sum;

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim3;) {
      int vl = vlmax;
      if (j + vlmax > dim3) vl = vsetvl_e32m1(dim3 - j);
      
      partial_sum = vfmv_v_f_f32m1(0, vl);

      for (int k = 0; k < dim2; k++) {
        vfloat32m1_t b_vec = vle32_v_f32m1(&mat_b[k * dim3 + j], vl);
        partial_sum = vfmacc_vf_f32m1(partial_sum, mat_a[i * dim2 + k], b_vec, vl);
      }

      // vfloat32m1_t bias_vec = vle32_v_f32m1(&bias[i * dim3 + j], vl);
      // partial_sum = vfadd_vv_f32m1(partial_sum, bias_vec, vl);
      vfloat32m1_t bias_vec = vle32_v_f32m1(&bias[j], vl);
      partial_sum = vfadd_vv_f32m1(partial_sum, bias_vec, vl);

      if (transposed == 0) vse32_v_f32m1(&o[i * dim3 + j], partial_sum, vl);
      else vsse32_v_f32m1(&o[j * dim1 + i], dim1 * 4, partial_sum, vl);
      j += vl;
    }
  }
}

// scale the matmul result
void matmul_scaled(float *mat_a, float *mat_b, float *o, float scale, int dim1, int dim2, int dim3, int transposed) {
  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t partial_sum;

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim3;) {
      int vl = vlmax;
      if (j + vlmax > dim3) vl = vsetvl_e32m1(dim3 - j);
      
      partial_sum = vfmv_v_f_f32m1(0, vl);

      for (int k = 0; k < dim2; k++) {
        vfloat32m1_t b_vec = vle32_v_f32m1(&mat_b[k * dim3 + j], vl);
        partial_sum = vfmacc_vf_f32m1(partial_sum, mat_a[i * dim2 + k], b_vec, vl);
      }

      partial_sum = vfdiv_vf_f32m1(partial_sum, scale, vl);

      if (transposed == 0) vse32_v_f32m1(&o[i * dim3 + j], partial_sum, vl);
      else vsse32_v_f32m1(&o[j * dim1 + i], dim1 * 4, partial_sum, vl);
      j += vl;
    }
  }
}

void transpose(float *mat, float *o, int row, int col) {
  size_t vlmax = vsetvlmax_e32m1();
  for (int i = 0; i < row;) {
    int vl = vlmax;
    if (i + vlmax > row) vl = vsetvl_e32m1(row - i);
    
    for (int j = 0; j < col; j++) {
      vfloat32m1_t vec = vle32_v_f32m1(&mat[i * col + j], vl);
      vsse32_v_f32m1(&o[j * row + i], 4 * row, vec, vl);
    }

    i += vl;
  }
}
