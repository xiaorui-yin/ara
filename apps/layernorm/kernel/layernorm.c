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

#ifdef VCD_DUMP
  event_trigger = +1
#endif

  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t mean, var;
  vfloat32m1_t tmp_vec;

  for (int i = 0; i < row;) {
    int vl = vlmax;
    // if (i + vlmax > row) vl = row - i;
    if (i + vlmax > row)
      vl = vsetvl_e32m1(row - i);

    // =================================================
    // mean calculation
    // =================================================
    mean = vfmv_v_f_f32m1(0, vl);

    for (int j = 0; j < col; j++) {
      // stride load: mat[i][j] ~ mat[i+vl][j]
      // NOTE: for strided store/load, the stride size is in byte (float32: 4
      // byte)
      vfloat32m1_t vec = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);

      // partial sum aggregation
      mean = vfadd_vv_f32m1(mean, vec, vl);
    }

    mean = vfdiv_vf_f32m1(mean, col, vl);

    // =================================================
    // variance calculation
    // =================================================
    var = vfmv_v_f_f32m1(0.00001, vl);

    for (int j = 0; j < col; j++) {
      vfloat32m1_t vec_x = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);

      tmp_vec = vfsub_vv_f32m1(vec_x, mean, vl);

      // store x(i) - mean(i)
      vsse32_v_f32m1(&mat[i * col + j], col * 4, tmp_vec, vl);

      tmp_vec = vfmul_vv_f32m1(tmp_vec, tmp_vec, vl);
      var = vfadd_vv_f32m1(var, tmp_vec, vl);
    }
    // Bessel's correction: divide by n - 1 instead of n!!!
    var = vfdiv_vf_f32m1(var, (col - 1), vl);
    // 1/sqrt(var)
    // TODO, vfrsqrt7 is not implemented
    // partial_sum = vfrsqrt7_v_f32m1(partial_sum, vl);
    var = vfsqrt_v_f32m1(var, vl);
    vfloat32m1_t vec_ones = vfmv_v_f_f32m1(1, vl);
    var = vfdiv_vv_f32m1(vec_ones, var, vl);

    // =================================================
    // apply to the elements
    // =================================================
    for (int j = 0; j < col; j++) {
      vfloat32m1_t vec_x = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);

      tmp_vec = vfmul_vv_f32m1(vec_x, var, vl);
      tmp_vec = vfmul_vf_f32m1(tmp_vec, alpha[j], vl);
      tmp_vec = vfadd_vf_f32m1(tmp_vec, beta[j], vl);
      vsse32_v_f32m1(&mat[i * col + j], col * 4, tmp_vec, vl);
    }

    i += vl;
  }

#ifdef VCD_DUMP
  event_trigger = -1
#endif
}

void layernorm_t(float *mat, float *alpha, float *beta, int row, int col) {

#ifdef VCD_DUMP
  event_trigger = +1
#endif

  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t mean, var;
  vfloat32m1_t tmp_vec;

  for (int i = 0; i < col;) {
    int vl = vlmax;
    if (i + vlmax > col)
      vl = vsetvl_e32m1(col - i);

    // =================================================
    // mean calculation
    // =================================================
    mean = vfmv_v_f_f32m1(0, vl);

    for (int j = 0; j < row; j++) {
      vfloat32m1_t vec = vle32_v_f32m1(&mat[j * col + i], vl);

      // partial sum aggregation
      mean = vfadd_vv_f32m1(mean, vec, vl);
    }

    mean = vfdiv_vf_f32m1(mean, row, vl);

    // =================================================
    // variance calculation
    // =================================================
    var = vfmv_v_f_f32m1(0.00001, vl);

    for (int j = 0; j < row; j++) {
      vfloat32m1_t vec_x = vle32_v_f32m1(&mat[j * col + i], vl);

      tmp_vec = vfsub_vv_f32m1(vec_x, mean, vl);

      // store x(i) - mean(i)
      vse32_v_f32m1(&mat[j * col + i], tmp_vec, vl);

      tmp_vec = vfmul_vv_f32m1(tmp_vec, tmp_vec, vl);
      var = vfadd_vv_f32m1(var, tmp_vec, vl);
    }
    // Bessel's correction: divide by n - 1 instead of n!!!
    var = vfdiv_vf_f32m1(var, (row - 1), vl);
    // 1/sqrt(var)
    // TODO, vfrsqrt7 is not implemented
    // var = vfrsqrt7_v_f32m1(var, vl);
    var = vfsqrt_v_f32m1(var, vl);
    vfloat32m1_t vec_ones = vfmv_v_f_f32m1(1, vl);
    var = vfdiv_vv_f32m1(vec_ones, var, vl);

    // =================================================
    // apply to the elements
    // =================================================
    for (int j = 0; j < row; j++) {
      vfloat32m1_t vec_x = vle32_v_f32m1(&mat[j * col + i], vl);

      tmp_vec = vfmul_vv_f32m1(vec_x, var, vl);
      tmp_vec = vfmul_vf_f32m1(tmp_vec, alpha[j], vl);
      tmp_vec = vfadd_vf_f32m1(tmp_vec, beta[j], vl);
      vse32_v_f32m1(&mat[j * col + i], tmp_vec, vl);
    }

    i += vl;
  }

#ifdef VCD_DUMP
  event_trigger = -1
#endif
}

void layernorm_v2(float *mat, float *alpha, float *beta, int row, int col) {
  float mean[row] __attribute__((aligned(32 * NR_LANES)));
  float var[row] __attribute__((aligned(32 * NR_LANES)));

  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t tmp_mean, tmp_var, tmp_vec;

  for (int i = 0; i < row;) {
    int vl = vlmax;
    // if (i + vlmax > row) vl = row - i;
    if (i + vlmax > row)
      vl = vsetvl_e32m1(row - i);

    // =================================================
    // mean and variance calculation
    // =================================================
    tmp_mean = vfmv_v_f_f32m1(0, vl);
    tmp_var = vfmv_v_f_f32m1(0.00001, vl);

    for (int j = 0; j < col; j++) {
      // stride load: mat[i][j] ~ mat[i+vl][j]
      // NOTE: for strided store/load, the stride size is in byte (float32: 4
      // byte)
      vfloat32m1_t vec = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);

      // partial sum aggregation
      // mean = (mean + vec) / current_length
      tmp_mean = vfadd_vv_f32m1(tmp_mean, vec, vl);
      tmp_mean = vfdiv_vf_f32m1(tmp_mean, j + 1, vl);

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
      vfloat32m1_t vec_var = vle32_v_f32m1(&var[i], vl);
      vfloat32m1_t vec_mean = vle32_v_f32m1(&mean[i], vl);
      vfloat32m1_t vec_x = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);
      // vfloat32m1_t vec_alpha = vlse32_v_f32m1(&alpha[i * col + j], col * 4,
      // vl); vfloat32m1_t vec_beta  = vlse32_v_f32m1(&beta[i * col + j], col *
      // 4, vl);

      tmp_vec = vfsub_vv_f32m1(vec_x, vec_mean, vl);
      tmp_vec = vfmul_vv_f32m1(vec_x, vec_var, vl);
      tmp_vec = vfmul_vf_f32m1(tmp_vec, alpha[i], vl);
      tmp_vec = vfadd_vf_f32m1(tmp_vec, beta[i], vl);
      vsse32_v_f32m1(&mat[i * col + j], col * 4, tmp_vec, vl);
    }

    i += vl;
  }
}
