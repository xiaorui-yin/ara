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

// mat_a: dim1 x dim2
// mat_b: dim2 x dim3
// o    : dim1 x dim3 (transposed dim3 x dim1)

/*
void matmul(float *mat_a, float *mat_b, float *o, int dim1, int dim2, int dim3,
int transposed) { size_t vlmax = vsetvlmax_e32m1(); vfloat32m1_t partial_sum;

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim3;) {
      int vl = vlmax;
      if (j + vlmax > dim3) vl = vsetvl_e32m1(dim3 - j);

      partial_sum = vfmv_v_f_f32m1(0, vl);

      for (int k = 0; k < dim2; k++) {
        vfloat32m1_t b_vec = vle32_v_f32m1(&mat_b[k * dim3 + j], vl);
        partial_sum = vfmacc_vf_f32m1(partial_sum, mat_a[i * dim2 + k], b_vec,
vl);
      }

      if (transposed == 0) vse32_v_f32m1(&o[i * dim3 + j], partial_sum, vl);
      else vsse32_v_f32m1(&o[j * dim1 + i], dim1 * 4, partial_sum, vl);
      j += vl;
    }
  }
}
*/

/*
void matmul(float *mat_a, float *mat_b, float *o, int dim1, int dim2, int dim3)
{ size_t vlmax = vsetvlmax_e32m1(); vfloat32m1_t partial_sum_1, partial_sum_2,
partial_sum_3, partial_sum_4, partial_sum_5, partial_sum_6, partial_sum_7,
partial_sum_8, partial_sum_9, partial_sum_10, partial_sum_11, partial_sum_12,
               partial_sum_13, partial_sum_14, partial_sum_15, partial_sum_16;

  for (int j = 0; j < dim3;) {
    int vl = vlmax;
    if (j + vlmax > dim3) vl = vsetvl_e32m1(dim3 - j);

    for (int i = 0; i < dim1; i = i + 16) {
      partial_sum_1 = vfmv_v_f_f32m1(0, vl);
      partial_sum_2 = vfmv_v_f_f32m1(0, vl);
      partial_sum_3 = vfmv_v_f_f32m1(0, vl);
      partial_sum_4 = vfmv_v_f_f32m1(0, vl);
      partial_sum_5 = vfmv_v_f_f32m1(0, vl);
      partial_sum_6 = vfmv_v_f_f32m1(0, vl);
      partial_sum_7 = vfmv_v_f_f32m1(0, vl);
      partial_sum_8 = vfmv_v_f_f32m1(0, vl);
      partial_sum_9 = vfmv_v_f_f32m1(0, vl);
      partial_sum_10 = vfmv_v_f_f32m1(0, vl);
      partial_sum_11 = vfmv_v_f_f32m1(0, vl);
      partial_sum_12 = vfmv_v_f_f32m1(0, vl);
      partial_sum_13 = vfmv_v_f_f32m1(0, vl);
      partial_sum_14 = vfmv_v_f_f32m1(0, vl);
      partial_sum_15 = vfmv_v_f_f32m1(0, vl);
      partial_sum_16 = vfmv_v_f_f32m1(0, vl);

      for (int k = 0; k < dim2; k += 2) {
        vfloat32m1_t b_vec = vle32_v_f32m1(&mat_b[k * dim3 + j], vl);
        partial_sum_1  = vfmacc_vf_f32m1(partial_sum_1,  mat_a[i * dim2 + k],
b_vec, vl); partial_sum_2  = vfmacc_vf_f32m1(partial_sum_2,  mat_a[(i + 1) *
dim2 + k],  b_vec, vl); vfloat32m1_t b_vec2 = vle32_v_f32m1(&mat_b[(k + 1) *
dim3 + j], vl); partial_sum_3  = vfmacc_vf_f32m1(partial_sum_3,  mat_a[(i + 2) *
dim2 + k],  b_vec, vl); partial_sum_4  = vfmacc_vf_f32m1(partial_sum_4, mat_a[(i
+ 3) * dim2 + k],  b_vec, vl); partial_sum_5  = vfmacc_vf_f32m1(partial_sum_5,
mat_a[(i + 4) * dim2 + k],  b_vec, vl); partial_sum_6  =
vfmacc_vf_f32m1(partial_sum_6,  mat_a[(i + 5) * dim2 + k],  b_vec, vl);
        partial_sum_7  = vfmacc_vf_f32m1(partial_sum_7,  mat_a[(i + 6) * dim2 +
k],  b_vec, vl); partial_sum_8  = vfmacc_vf_f32m1(partial_sum_8,  mat_a[(i + 7)
* dim2 + k],  b_vec, vl); partial_sum_9  = vfmacc_vf_f32m1(partial_sum_9,
mat_a[(i + 8) * dim2 + k],  b_vec, vl); partial_sum_10 =
vfmacc_vf_f32m1(partial_sum_10, mat_a[(i + 9) * dim2 + k],  b_vec, vl);
        partial_sum_11 = vfmacc_vf_f32m1(partial_sum_11, mat_a[(i + 10) * dim2 +
k], b_vec, vl); partial_sum_12 = vfmacc_vf_f32m1(partial_sum_12, mat_a[(i + 11)
* dim2 + k], b_vec, vl); partial_sum_13 = vfmacc_vf_f32m1(partial_sum_13,
mat_a[(i + 12) * dim2 + k], b_vec, vl); partial_sum_14 =
vfmacc_vf_f32m1(partial_sum_14, mat_a[(i + 13) * dim2 + k], b_vec, vl);
        partial_sum_15 = vfmacc_vf_f32m1(partial_sum_15, mat_a[(i + 14) * dim2 +
k], b_vec, vl); partial_sum_16 = vfmacc_vf_f32m1(partial_sum_16, mat_a[(i + 15)
* dim2 + k], b_vec, vl);

        partial_sum_1  = vfmacc_vf_f32m1(partial_sum_1,  mat_a[i * dim2 + k +
1],        b_vec2, vl); partial_sum_2  = vfmacc_vf_f32m1(partial_sum_2, mat_a[(i
+ 1) * dim2 + k + 1],  b_vec2, vl); partial_sum_3  =
vfmacc_vf_f32m1(partial_sum_3,  mat_a[(i + 2) * dim2 + k + 1],  b_vec2, vl);
        partial_sum_4  = vfmacc_vf_f32m1(partial_sum_4,  mat_a[(i + 3) * dim2 +
k + 1],  b_vec2, vl); partial_sum_5  = vfmacc_vf_f32m1(partial_sum_5,  mat_a[(i
+ 4) * dim2 + k + 1],  b_vec2, vl); partial_sum_6  =
vfmacc_vf_f32m1(partial_sum_6,  mat_a[(i + 5) * dim2 + k + 1],  b_vec2, vl);
        partial_sum_7  = vfmacc_vf_f32m1(partial_sum_7,  mat_a[(i + 6) * dim2 +
k + 1],  b_vec2, vl); partial_sum_8  = vfmacc_vf_f32m1(partial_sum_8,  mat_a[(i
+ 7) * dim2 + k + 1],  b_vec2, vl); partial_sum_9  =
vfmacc_vf_f32m1(partial_sum_9,  mat_a[(i + 8) * dim2 + k + 1],  b_vec2, vl);
        partial_sum_10 = vfmacc_vf_f32m1(partial_sum_10, mat_a[(i + 9) * dim2 +
k + 1],  b_vec2, vl); partial_sum_11 = vfmacc_vf_f32m1(partial_sum_11, mat_a[(i
+ 10) * dim2 + k + 1], b_vec2, vl); partial_sum_12 =
vfmacc_vf_f32m1(partial_sum_12, mat_a[(i + 11) * dim2 + k + 1], b_vec2, vl);
        partial_sum_13 = vfmacc_vf_f32m1(partial_sum_13, mat_a[(i + 12) * dim2 +
k + 1], b_vec2, vl); partial_sum_14 = vfmacc_vf_f32m1(partial_sum_14, mat_a[(i +
13) * dim2 + k + 1], b_vec2, vl); partial_sum_15 =
vfmacc_vf_f32m1(partial_sum_15, mat_a[(i + 14) * dim2 + k + 1], b_vec2, vl);
        partial_sum_16 = vfmacc_vf_f32m1(partial_sum_16, mat_a[(i + 15) * dim2 +
k + 1], b_vec2, vl);
      }

      vse32_v_f32m1(&o[i * dim3 + j], partial_sum_1, vl);
      vse32_v_f32m1(&o[(i + 1) * dim3 + j], partial_sum_2, vl);
      vse32_v_f32m1(&o[(i + 2) * dim3 + j], partial_sum_3, vl);
      vse32_v_f32m1(&o[(i + 3) * dim3 + j], partial_sum_4, vl);
      vse32_v_f32m1(&o[(i + 4) * dim3 + j], partial_sum_5, vl);
      vse32_v_f32m1(&o[(i + 5) * dim3 + j], partial_sum_6, vl);
      vse32_v_f32m1(&o[(i + 6) * dim3 + j], partial_sum_7, vl);
      vse32_v_f32m1(&o[(i + 7) * dim3 + j], partial_sum_8, vl);
      vse32_v_f32m1(&o[(i + 8) * dim3 + j], partial_sum_9, vl);
      vse32_v_f32m1(&o[(i + 9) * dim3 + j], partial_sum_10, vl);
      vse32_v_f32m1(&o[(i + 10) * dim3 + j], partial_sum_11, vl);
      vse32_v_f32m1(&o[(i + 11) * dim3 + j], partial_sum_12, vl);
      vse32_v_f32m1(&o[(i + 12) * dim3 + j], partial_sum_13, vl);
      vse32_v_f32m1(&o[(i + 13) * dim3 + j], partial_sum_14, vl);
      vse32_v_f32m1(&o[(i + 14) * dim3 + j], partial_sum_15, vl);
      vse32_v_f32m1(&o[(i + 15) * dim3 + j], partial_sum_16, vl);
    }
    j += vl;
  }
}
*/

/*
void matmul(float *mat_a, float *mat_b, float *o, int dim1, int dim2, int dim3,
int transposed) { size_t vlmax = vsetvlmax_e32m1(); vfloat32m1_t partial_sum_1,
partial_sum_2, partial_sum_3, partial_sum_4;

  for (int j = 0; j < dim3;) {
    int vl = vlmax;
    if (j + vlmax > dim3) vl = vsetvl_e32m1(dim3 - j);

    for (int i = 0; i < dim1; i = i + 4) {
      partial_sum_1 = vfmv_v_f_f32m1(0, vl);
      partial_sum_2 = vfmv_v_f_f32m1(0, vl);
      partial_sum_3 = vfmv_v_f_f32m1(0, vl);
      partial_sum_4 = vfmv_v_f_f32m1(0, vl);

      int b_vec_ptr = mat_b + j;
      vfloat32m1_t b_vec_1, b_vec_2;
      b_vec_1 = vle32_v_f32m1(b_vec_ptr, vl);
      for (int k = 0; k < dim2; k += 2) {
        vfloat32m1_t b_vec_1 = vle32_v_f32m1(&mat_b[k * dim3 + j], vl);
        partial_sum_1  = vfmacc_vf_f32m1(partial_sum_1,  mat_a[i * dim2 + k],
b_vec_1, vl); partial_sum_2  = vfmacc_vf_f32m1(partial_sum_2,  mat_a[(i + 1) *
dim2 + k],  b_vec_1, vl); vfloat32m1_t b_vec_2 = vle32_v_f32m1(&mat_b[(k + 1) *
dim3 + j], vl); partial_sum_3  = vfmacc_vf_f32m1(partial_sum_3,  mat_a[(i + 2) *
dim2 + k],  b_vec_1, vl); partial_sum_4  = vfmacc_vf_f32m1(partial_sum_4,
mat_a[(i + 3) * dim2 + k],  b_vec_1, vl);

        partial_sum_1  = vfmacc_vf_f32m1(partial_sum_1,  mat_a[i * dim2 + k +
1],        b_vec_2, vl); partial_sum_2  = vfmacc_vf_f32m1(partial_sum_2,
mat_a[(i + 1) * dim2 + k + 1],  b_vec_2, vl); partial_sum_3  =
vfmacc_vf_f32m1(partial_sum_3,  mat_a[(i + 2) * dim2 + k + 1],  b_vec_2, vl);
        partial_sum_4  = vfmacc_vf_f32m1(partial_sum_4,  mat_a[(i + 3) * dim2 +
k + 1],  b_vec_2, vl);
      }

      vse32_v_f32m1(&o[i * dim3 + j], partial_sum_1, vl);
      vse32_v_f32m1(&o[(i + 1) * dim3 + j], partial_sum_2, vl);
      vse32_v_f32m1(&o[(i + 2) * dim3 + j], partial_sum_3, vl);
      vse32_v_f32m1(&o[(i + 3) * dim3 + j], partial_sum_4, vl);
    }
    j += vl;
  }
}
*/

/*
void matmul(float *mat_a, float *mat_b, float *o, int dim1, int dim2, int dim3,
int transposed) { size_t vlmax = vsetvlmax_e32m1(); vfloat32m1_t partial_sum_1,
partial_sum_2, partial_sum_3, partial_sum_4, partial_sum_5, partial_sum_6,
partial_sum_7, partial_sum_8, partial_sum_9, partial_sum_10, partial_sum_11,
partial_sum_12, partial_sum_13, partial_sum_14, partial_sum_15, partial_sum_16;

  for (int j = 0; j < dim3;) {
    int vl = vlmax;
    if (j + vlmax > dim3) vl = vsetvl_e32m1(dim3 - j);

    for (int i = 0; i < dim1; i = i + 16) {
      partial_sum_1 = vfmv_v_f_f32m1(0, vl);
      partial_sum_2 = vfmv_v_f_f32m1(0, vl);
      partial_sum_3 = vfmv_v_f_f32m1(0, vl);
      partial_sum_4 = vfmv_v_f_f32m1(0, vl);
      partial_sum_5 = vfmv_v_f_f32m1(0, vl);
      partial_sum_6 = vfmv_v_f_f32m1(0, vl);
      partial_sum_7 = vfmv_v_f_f32m1(0, vl);
      partial_sum_8 = vfmv_v_f_f32m1(0, vl);
      partial_sum_9 = vfmv_v_f_f32m1(0, vl);
      partial_sum_10 = vfmv_v_f_f32m1(0, vl);
      partial_sum_11 = vfmv_v_f_f32m1(0, vl);
      partial_sum_12 = vfmv_v_f_f32m1(0, vl);
      partial_sum_13 = vfmv_v_f_f32m1(0, vl);
      partial_sum_14 = vfmv_v_f_f32m1(0, vl);
      partial_sum_15 = vfmv_v_f_f32m1(0, vl);
      partial_sum_16 = vfmv_v_f_f32m1(0, vl);

      vfloat32m1_t b_vec, b_vec2;
      for (int k = 0; k < dim2; k += 2) {
        b_vec = vle32_v_f32m1(&mat_b[k * dim3 + j], vl);
        partial_sum_1  = vfmacc_vf_f32m1(partial_sum_1,  mat_a[i * dim2 + k],
b_vec, vl); partial_sum_2  = vfmacc_vf_f32m1(partial_sum_2,  mat_a[(i + 1) *
dim2 + k],  b_vec, vl); b_vec2 = vle32_v_f32m1(&mat_b[(k + 1) * dim3 + j], vl);
        partial_sum_3  = vfmacc_vf_f32m1(partial_sum_3,  mat_a[(i + 2) * dim2 +
k],  b_vec, vl); partial_sum_4  = vfmacc_vf_f32m1(partial_sum_4,  mat_a[(i + 3)
* dim2 + k],  b_vec, vl); partial_sum_5  = vfmacc_vf_f32m1(partial_sum_5,
mat_a[(i + 4) * dim2 + k],  b_vec, vl); partial_sum_6  =
vfmacc_vf_f32m1(partial_sum_6,  mat_a[(i + 5) * dim2 + k],  b_vec, vl);
        partial_sum_7  = vfmacc_vf_f32m1(partial_sum_7,  mat_a[(i + 6) * dim2 +
k],  b_vec, vl); partial_sum_8  = vfmacc_vf_f32m1(partial_sum_8,  mat_a[(i + 7)
* dim2 + k],  b_vec, vl); partial_sum_9  = vfmacc_vf_f32m1(partial_sum_9,
mat_a[(i + 8) * dim2 + k],  b_vec, vl); partial_sum_10 =
vfmacc_vf_f32m1(partial_sum_10, mat_a[(i + 9) * dim2 + k],  b_vec, vl);
        partial_sum_11 = vfmacc_vf_f32m1(partial_sum_11, mat_a[(i + 10) * dim2 +
k], b_vec, vl); partial_sum_12 = vfmacc_vf_f32m1(partial_sum_12, mat_a[(i + 11)
* dim2 + k], b_vec, vl); partial_sum_13 = vfmacc_vf_f32m1(partial_sum_13,
mat_a[(i + 12) * dim2 + k], b_vec, vl); partial_sum_14 =
vfmacc_vf_f32m1(partial_sum_14, mat_a[(i + 13) * dim2 + k], b_vec, vl);
        partial_sum_15 = vfmacc_vf_f32m1(partial_sum_15, mat_a[(i + 14) * dim2 +
k], b_vec, vl); partial_sum_16 = vfmacc_vf_f32m1(partial_sum_16, mat_a[(i + 15)
* dim2 + k], b_vec, vl);

        // if this is the last computation, leave it to the store stage
        if (k < dim2 - 2) {
          partial_sum_1  = vfmacc_vf_f32m1(partial_sum_1,  mat_a[i * dim2 + k +
1],        b_vec2, vl); partial_sum_2  = vfmacc_vf_f32m1(partial_sum_2, mat_a[(i
+ 1) * dim2 + k + 1],  b_vec2, vl); partial_sum_3  =
vfmacc_vf_f32m1(partial_sum_3,  mat_a[(i + 2) * dim2 + k + 1],  b_vec2, vl);
          partial_sum_4  = vfmacc_vf_f32m1(partial_sum_4,  mat_a[(i + 3) * dim2
+ k + 1],  b_vec2, vl); partial_sum_5  = vfmacc_vf_f32m1(partial_sum_5, mat_a[(i
+ 4) * dim2 + k + 1],  b_vec2, vl); partial_sum_6  =
vfmacc_vf_f32m1(partial_sum_6,  mat_a[(i + 5) * dim2 + k + 1],  b_vec2, vl);
          partial_sum_7  = vfmacc_vf_f32m1(partial_sum_7,  mat_a[(i + 6) * dim2
+ k + 1],  b_vec2, vl); partial_sum_8  = vfmacc_vf_f32m1(partial_sum_8, mat_a[(i
+ 7) * dim2 + k + 1],  b_vec2, vl); partial_sum_9  =
vfmacc_vf_f32m1(partial_sum_9,  mat_a[(i + 8) * dim2 + k + 1],  b_vec2, vl);
          partial_sum_10 = vfmacc_vf_f32m1(partial_sum_10, mat_a[(i + 9) * dim2
+ k + 1],  b_vec2, vl); partial_sum_11 = vfmacc_vf_f32m1(partial_sum_11,
mat_a[(i + 10) * dim2 + k + 1], b_vec2, vl); partial_sum_12 =
vfmacc_vf_f32m1(partial_sum_12, mat_a[(i + 11) * dim2 + k + 1], b_vec2, vl);
          partial_sum_13 = vfmacc_vf_f32m1(partial_sum_13, mat_a[(i + 12) * dim2
+ k + 1], b_vec2, vl); partial_sum_14 = vfmacc_vf_f32m1(partial_sum_14, mat_a[(i
+ 13) * dim2 + k + 1], b_vec2, vl); partial_sum_15 =
vfmacc_vf_f32m1(partial_sum_15, mat_a[(i + 14) * dim2 + k + 1], b_vec2, vl);
          partial_sum_16 = vfmacc_vf_f32m1(partial_sum_16, mat_a[(i + 15) * dim2
+ k + 1], b_vec2, vl);
        }
      }

      // increase utilization, otherwise, during store stage, the FPU is idle
      // mat_a[i * dim2 + dim2 - 1]: the last element in i-th row of matrix a
      partial_sum_1  = vfmacc_vf_f32m1(partial_sum_1,  mat_a[i * dim2 + dim2 -
1], b_vec2, vl); vse32_v_f32m1(&o[i * dim3 + j], partial_sum_1, vl);
      partial_sum_2  = vfmacc_vf_f32m1(partial_sum_2,  mat_a[(i + 1) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 1) * dim3 + j], partial_sum_2,
vl); partial_sum_3  = vfmacc_vf_f32m1(partial_sum_3,  mat_a[(i + 2) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 2) * dim3 + j], partial_sum_3,
vl); partial_sum_4  = vfmacc_vf_f32m1(partial_sum_4,  mat_a[(i + 3) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 3) * dim3 + j], partial_sum_4,
vl); partial_sum_5  = vfmacc_vf_f32m1(partial_sum_5,  mat_a[(i + 4) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 4) * dim3 + j], partial_sum_5,
vl); partial_sum_6  = vfmacc_vf_f32m1(partial_sum_6,  mat_a[(i + 5) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 5) * dim3 + j], partial_sum_6,
vl); partial_sum_7  = vfmacc_vf_f32m1(partial_sum_7,  mat_a[(i + 6) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 6) * dim3 + j], partial_sum_7,
vl); partial_sum_8  = vfmacc_vf_f32m1(partial_sum_8,  mat_a[(i + 7) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 7) * dim3 + j], partial_sum_8,
vl); partial_sum_9  = vfmacc_vf_f32m1(partial_sum_9,  mat_a[(i + 8) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 8) * dim3 + j], partial_sum_9,
vl); partial_sum_10  = vfmacc_vf_f32m1(partial_sum_10,  mat_a[(i + 9) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 9) * dim3 + j], partial_sum_10,
vl); partial_sum_11  = vfmacc_vf_f32m1(partial_sum_11,  mat_a[(i + 10) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 10) * dim3 + j], partial_sum_11,
vl); partial_sum_12  = vfmacc_vf_f32m1(partial_sum_12,  mat_a[(i + 11) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 11) * dim3 + j], partial_sum_12,
vl); partial_sum_13  = vfmacc_vf_f32m1(partial_sum_13,  mat_a[(i + 12) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 12) * dim3 + j], partial_sum_13,
vl); partial_sum_14  = vfmacc_vf_f32m1(partial_sum_14,  mat_a[(i + 13) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 13) * dim3 + j], partial_sum_14,
vl); partial_sum_15  = vfmacc_vf_f32m1(partial_sum_15,  mat_a[(i + 14) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 14) * dim3 + j], partial_sum_15,
vl); partial_sum_16  = vfmacc_vf_f32m1(partial_sum_16,  mat_a[(i + 15) * dim2 +
dim2 - 1],  b_vec2, vl); vse32_v_f32m1(&o[(i + 15) * dim3 + j], partial_sum_16,
vl);
    }
    j += vl;
  }
}
*/

void matmul(float *mat_a, float *mat_b, float *o, int dim1, int dim2, int dim3,
            int transposed) {
  size_t vlmax = vsetvlmax_e32m4();
  vfloat32m4_t partial_sum_1, partial_sum_2, partial_sum_3, partial_sum_4,
      partial_sum_5, partial_sum_6, partial_sum_7, partial_sum_8, partial_sum_9,
      partial_sum_10, partial_sum_11, partial_sum_12, partial_sum_13,
      partial_sum_14, partial_sum_15, partial_sum_16;

  for (int j = 0; j < dim3;) {
    int vl = vlmax;
    if (j + vlmax > dim3)
      vl = vsetvl_e32m4(dim3 - j);

    for (int i = 0; i < dim1; i = i + 4) {
      partial_sum_1 = vfmv_v_f_f32m4(0, vl);
      partial_sum_2 = vfmv_v_f_f32m4(0, vl);
      partial_sum_3 = vfmv_v_f_f32m4(0, vl);
      partial_sum_4 = vfmv_v_f_f32m4(0, vl);
      partial_sum_5 = vfmv_v_f_f32m4(0, vl);
      partial_sum_6 = vfmv_v_f_f32m4(0, vl);
      partial_sum_7 = vfmv_v_f_f32m4(0, vl);
      partial_sum_8 = vfmv_v_f_f32m4(0, vl);
      partial_sum_9 = vfmv_v_f_f32m4(0, vl);
      partial_sum_10 = vfmv_v_f_f32m4(0, vl);
      partial_sum_11 = vfmv_v_f_f32m4(0, vl);
      partial_sum_12 = vfmv_v_f_f32m4(0, vl);
      partial_sum_13 = vfmv_v_f_f32m4(0, vl);
      partial_sum_14 = vfmv_v_f_f32m4(0, vl);
      partial_sum_15 = vfmv_v_f_f32m4(0, vl);
      partial_sum_16 = vfmv_v_f_f32m4(0, vl);

      vfloat32m4_t b_vec, b_vec_2;
      int k = 0;
      float *b_ptr = mat_b + j;
      float *a_ptr = mat_a + i * dim2;
      b_vec = vle32_v_f32m4(b_ptr, vl);
      b_ptr += dim3;
      while (k != dim2) {
        partial_sum_1 = vfmacc_vf_f32m4(partial_sum_1, *(a_ptr + k), b_vec, vl);
        b_vec_2 = vle32_v_f32m4(b_ptr, vl);
        b_ptr += dim3;
        partial_sum_2 =
            vfmacc_vf_f32m4(partial_sum_2, *(a_ptr + dim2 + k), b_vec, vl);
        partial_sum_3 = vfmacc_vf_f32m4(partial_sum_3,
                                        *(a_ptr + (2 * dim2) + k), b_vec, vl);
        partial_sum_4 = vfmacc_vf_f32m4(partial_sum_4,
                                        *(a_ptr + (3 * dim2) + k), b_vec, vl);
        partial_sum_5 = vfmacc_vf_f32m4(partial_sum_5,
                                        *(a_ptr + (4 * dim2) + k), b_vec, vl);
        partial_sum_6 = vfmacc_vf_f32m4(partial_sum_6,
                                        *(a_ptr + (5 * dim2) + k), b_vec, vl);
        partial_sum_7 = vfmacc_vf_f32m4(partial_sum_7,
                                        *(a_ptr + (6 * dim2) + k), b_vec, vl);
        partial_sum_8 = vfmacc_vf_f32m4(partial_sum_8,
                                        *(a_ptr + (7 * dim2) + k), b_vec, vl);
        partial_sum_9 = vfmacc_vf_f32m4(partial_sum_9,
                                        *(a_ptr + (8 * dim2) + k), b_vec, vl);
        partial_sum_10 = vfmacc_vf_f32m4(partial_sum_10,
                                         *(a_ptr + (9 * dim2) + k), b_vec, vl);
        partial_sum_11 = vfmacc_vf_f32m4(partial_sum_11,
                                         *(a_ptr + (10 * dim2) + k), b_vec, vl);
        partial_sum_12 = vfmacc_vf_f32m4(partial_sum_12,
                                         *(a_ptr + (11 * dim2) + k), b_vec, vl);
        partial_sum_13 = vfmacc_vf_f32m4(partial_sum_13,
                                         *(a_ptr + (12 * dim2) + k), b_vec, vl);
        partial_sum_14 = vfmacc_vf_f32m4(partial_sum_14,
                                         *(a_ptr + (13 * dim2) + k), b_vec, vl);
        partial_sum_15 = vfmacc_vf_f32m4(partial_sum_15,
                                         *(a_ptr + (14 * dim2) + k), b_vec, vl);
        partial_sum_16 = vfmacc_vf_f32m4(partial_sum_16,
                                         *(a_ptr + (15 * dim2) + k), b_vec, vl);

        k++;

        // if this is the last computation, leave it to the store stage
        if (k == dim2 - 1)
          break;

        partial_sum_1 =
            vfmacc_vf_f32m4(partial_sum_1, *(a_ptr + k), b_vec_2, vl);
        b_vec = vle32_v_f32m4(b_ptr, vl);
        b_ptr += dim3;
        partial_sum_2 =
            vfmacc_vf_f32m4(partial_sum_2, *(a_ptr + dim2 + k), b_vec_2, vl);
        partial_sum_3 = vfmacc_vf_f32m4(partial_sum_3,
                                        *(a_ptr + (2 * dim2) + k), b_vec_2, vl);
        partial_sum_4 = vfmacc_vf_f32m4(partial_sum_4,
                                        *(a_ptr + (3 * dim2) + k), b_vec_2, vl);
        partial_sum_5 = vfmacc_vf_f32m4(partial_sum_5,
                                        *(a_ptr + (4 * dim2) + k), b_vec_2, vl);
        partial_sum_6 = vfmacc_vf_f32m4(partial_sum_6,
                                        *(a_ptr + (5 * dim2) + k), b_vec_2, vl);
        partial_sum_7 = vfmacc_vf_f32m4(partial_sum_7,
                                        *(a_ptr + (6 * dim2) + k), b_vec_2, vl);
        partial_sum_8 = vfmacc_vf_f32m4(partial_sum_8,
                                        *(a_ptr + (7 * dim2) + k), b_vec_2, vl);
        partial_sum_9 = vfmacc_vf_f32m4(partial_sum_9,
                                        *(a_ptr + (8 * dim2) + k), b_vec_2, vl);
        partial_sum_10 = vfmacc_vf_f32m4(
            partial_sum_10, *(a_ptr + (9 * dim2) + k), b_vec_2, vl);
        partial_sum_11 = vfmacc_vf_f32m4(
            partial_sum_11, *(a_ptr + (10 * dim2) + k), b_vec_2, vl);
        partial_sum_12 = vfmacc_vf_f32m4(
            partial_sum_12, *(a_ptr + (11 * dim2) + k), b_vec_2, vl);
        partial_sum_13 = vfmacc_vf_f32m4(
            partial_sum_13, *(a_ptr + (12 * dim2) + k), b_vec_2, vl);
        partial_sum_14 = vfmacc_vf_f32m4(
            partial_sum_14, *(a_ptr + (13 * dim2) + k), b_vec_2, vl);
        partial_sum_15 = vfmacc_vf_f32m4(
            partial_sum_15, *(a_ptr + (14 * dim2) + k), b_vec_2, vl);
        partial_sum_16 = vfmacc_vf_f32m4(
            partial_sum_16, *(a_ptr + (15 * dim2) + k), b_vec_2, vl);

        k++;
      }

      // increase utilization, otherwise, during store stage, the FPU is idle
      // mat_a[i * dim2 + dim2 - 1]: the last element in i-th row of matrix a
      partial_sum_1 = vfmacc_vf_f32m4(partial_sum_1, *(a_ptr + k), b_vec_2, vl);
      vse32_v_f32m4(&o[i * dim3 + j], partial_sum_1, vl);
      partial_sum_2 =
          vfmacc_vf_f32m4(partial_sum_2, *(a_ptr + dim2 + k), b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 1) * dim3 + j], partial_sum_2, vl);
      partial_sum_3 = vfmacc_vf_f32m4(partial_sum_3, *(a_ptr + (2 * dim2) + k),
                                      b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 2) * dim3 + j], partial_sum_3, vl);
      partial_sum_4 = vfmacc_vf_f32m4(partial_sum_4, *(a_ptr + (3 * dim2) + k),
                                      b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 3) * dim3 + j], partial_sum_4, vl);
      partial_sum_5 = vfmacc_vf_f32m4(partial_sum_5, *(a_ptr + (4 * dim2) + k),
                                      b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 4) * dim3 + j], partial_sum_5, vl);
      partial_sum_6 = vfmacc_vf_f32m4(partial_sum_6, *(a_ptr + (5 * dim2) + k),
                                      b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 5) * dim3 + j], partial_sum_6, vl);
      partial_sum_7 = vfmacc_vf_f32m4(partial_sum_7, *(a_ptr + (6 * dim2) + k),
                                      b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 6) * dim3 + j], partial_sum_7, vl);
      partial_sum_8 = vfmacc_vf_f32m4(partial_sum_8, *(a_ptr + (7 * dim2) + k),
                                      b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 7) * dim3 + j], partial_sum_8, vl);
      partial_sum_9 = vfmacc_vf_f32m4(partial_sum_9, *(a_ptr + (8 * dim2) + k),
                                      b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 8) * dim3 + j], partial_sum_9, vl);
      partial_sum_10 = vfmacc_vf_f32m4(partial_sum_10,
                                       *(a_ptr + (9 * dim2) + k), b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 9) * dim3 + j], partial_sum_10, vl);
      partial_sum_11 = vfmacc_vf_f32m4(partial_sum_11,
                                       *(a_ptr + (10 * dim2) + k), b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 10) * dim3 + j], partial_sum_11, vl);
      partial_sum_12 = vfmacc_vf_f32m4(partial_sum_12,
                                       *(a_ptr + (11 * dim2) + k), b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 11) * dim3 + j], partial_sum_12, vl);
      partial_sum_13 = vfmacc_vf_f32m4(partial_sum_13,
                                       *(a_ptr + (12 * dim2) + k), b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 12) * dim3 + j], partial_sum_13, vl);
      partial_sum_14 = vfmacc_vf_f32m4(partial_sum_14,
                                       *(a_ptr + (13 * dim2) + k), b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 13) * dim3 + j], partial_sum_14, vl);
      partial_sum_15 = vfmacc_vf_f32m4(partial_sum_15,
                                       *(a_ptr + (14 * dim2) + k), b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 14) * dim3 + j], partial_sum_15, vl);
      partial_sum_16 = vfmacc_vf_f32m4(partial_sum_16,
                                       *(a_ptr + (15 * dim2) + k), b_vec_2, vl);
      vse32_v_f32m4(&o[(i + 15) * dim3 + j], partial_sum_16, vl);
    }
    j += vl;
  }
}
