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

#include "riscv_vector.h"

void layernorm(float *mat, float *alpha, float *beta, int row, int col) {
  float mean[row] __attribute__((aligned(32 * NR_LANES)));
  float var[row] __attribute__((aligned(32 * NR_LANES)));

  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t partial_sum;
  vfloat32m1_t tmp_vec;

  for (int i = 0; i < row;) {
    int vl = vlmax;
    //if (i + vlmax > row) vl = row - i;
    if (i + vlmax > row) vl = vsetvl_e32m1(row - i);

    // =================================================
    // mean calculation
    // =================================================
    partial_sum = vfmv_v_f_f32m1(0, vl);

    for (int j = 0; j < col; j++) {
      // stride load: mat[i][j] ~ mat[i+vl][j]
      // NOTE: for strided store/load, the stride size is in byte (float32: 4 byte)
      vfloat32m1_t vec = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);

      // partial sum aggregation
      partial_sum = vfadd_vv_f32m1(partial_sum, vec, vl);
    }

    partial_sum = vfdiv_vf_f32m1(partial_sum, col, vl);
    vse32_v_f32m1(&mean[i], partial_sum, vl);

    // =================================================
    // variance calculation
    // =================================================
    partial_sum = vfmv_v_f_f32m1(0.00001, vl);
    //partial_sum = vfmv_v_f_f32m1(0, vl);

    for (int j = 0; j < col; j++) {
      vfloat32m1_t vec_x = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);
      vfloat32m1_t vec_mean = vle32_v_f32m1(&mean[i], vl);

      tmp_vec = vfsub_vv_f32m1(vec_x, vec_mean, vl);

      // store x(i) - mean(i)
      vsse32_v_f32m1(&mat[i * col + j], col * 4, tmp_vec, vl);

      tmp_vec = vfmul_vv_f32m1(tmp_vec, tmp_vec, vl);
      partial_sum = vfadd_vv_f32m1(partial_sum, tmp_vec, vl);
    }
    // Bessel's correction: divide by n - 1 instead of n!!!
    partial_sum = vfdiv_vf_f32m1(partial_sum, (col - 1), vl);
    // 1/sqrt(var)
    // TODO, vfrsqrt7 is not implemented
    // partial_sum = vfrsqrt7_v_f32m1(partial_sum, vl);
    partial_sum = vfsqrt_v_f32m1(partial_sum, vl);
    vfloat32m1_t vec_ones = vfmv_v_f_f32m1(1, vl);
    partial_sum = vfdiv_vv_f32m1(vec_ones, partial_sum, vl);

    vse32_v_f32m1(&var[i], partial_sum, vl);

    // =================================================
    // apply to the elements
    // =================================================
    for (int j = 0; j < col; j++) {
      vfloat32m1_t vec_var   = vle32_v_f32m1(&var[i], vl);
      vfloat32m1_t vec_x     = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);
      // vfloat32m1_t vec_alpha = vlse32_v_f32m1(&alpha[i * col + j], col * 4, vl);
      // vfloat32m1_t vec_beta  = vlse32_v_f32m1(&beta[i * col + j], col * 4, vl);

      tmp_vec = vfmul_vv_f32m1(vec_x, vec_var, vl);
      // tmp_vec = vfmul_vv_f32m1(tmp_vec, vec_alpha, vl);
      // tmp_vec = vfadd_vv_f32m1(tmp_vec, vec_beta, vl);
      tmp_vec = vfmul_vf_f32m1(tmp_vec, alpha[j], vl);
      tmp_vec = vfadd_vf_f32m1(tmp_vec, beta[j], vl);
      vsse32_v_f32m1(&mat[i * col + j], col * 4, tmp_vec, vl);
    }

    i += vl;
  }
}

void layernorm_v2(float *mat, float *alpha, float *beta, int row, int col) {
  float mean[row] __attribute__((aligned(32 * NR_LANES)));
  float var[row] __attribute__((aligned(32 * NR_LANES)));

  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t tmp_mean, tmp_var, tmp_vec;

  for (int i = 0; i < row;) {
    int vl = vlmax;
    //if (i + vlmax > row) vl = row - i;
    if (i + vlmax > row) vl = vsetvl_e32m1(row - i);

    // =================================================
    // mean and variance calculation
    // =================================================
    tmp_mean = vfmv_v_f_f32m1(0, vl);
    tmp_var = vfmv_v_f_f32m1(0.00001, vl);

    for (int j = 0; j < col; j++) {
      // stride load: mat[i][j] ~ mat[i+vl][j]
      // NOTE: for strided store/load, the stride size is in byte (float32: 4 byte)
      vfloat32m1_t vec = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);

      // partial sum aggregation
      // mean = (mean + vec) / current_length
      tmp_mean = vfadd_vv_f32m1(tmp_mean, vec, vl);
      tmp_mean = vfdiv_vf_f32m1(tmp_mean, j+1, vl);

      // var = var + (x - current_mean)^2
      tmp_vec = vfsub_vv_f32m1(vec, tmp_mean, vl);
      tmp_var = vfmacc_vv_f32m1(tmp_var, tmp_vec, tmp_vec, vl);
    }

    // store mean
    vse32_v_f32m1(&mean[i], tmp_mean, vl);

    // for a set of data, divide by n - 1 instead of n!!!
    tmp_var = vfdiv_vf_f32m1(tmp_var, (col - 1), vl);
    // 1/sqrt(var)
    tmp_var = vfrsqrt7_v_f32m1(tmp_var, vl);
    // store standard deviation
    vse32_v_f32m1(&var[i], tmp_var, vl);

    // =================================================
    // apply to the elements
    // =================================================
    for (int j = 0; j < col; j++) {
      vfloat32m1_t vec_var   = vle32_v_f32m1(&var[i], vl);
      vfloat32m1_t vec_mean  = vle32_v_f32m1(&mean[i], vl);
      vfloat32m1_t vec_x     = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);
      // vfloat32m1_t vec_alpha = vlse32_v_f32m1(&alpha[i * col + j], col * 4, vl);
      // vfloat32m1_t vec_beta  = vlse32_v_f32m1(&beta[i * col + j], col * 4, vl);

      tmp_vec = vfsub_vv_f32m1(vec_x, vec_mean, vl);
      tmp_vec = vfmul_vv_f32m1(vec_x, vec_var, vl);
      tmp_vec = vfmul_vf_f32m1(tmp_vec, alpha[i], vl);
      tmp_vec = vfadd_vf_f32m1(tmp_vec, beta[i], vl);
      vsse32_v_f32m1(&mat[i * col + j], col * 4, tmp_vec, vl);
    }

    i += vl;
  }
}
