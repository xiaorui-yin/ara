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

// Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers

#include "riscv_vector.h"
#include "exp.h"

void softmax(float *mat, int row, int col) {
  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t cur_max, next_max, cur_sum, next_sum;
  vfloat32m1_t tmp_vec;

  for (int i = 0; i < row;) {
    int vl = vlmax;
    if (i + vlmax > row) vl = vsetvl_e32m1(row - i);

    // =================================================
    // update max and sum
    // =================================================

    // initialize cur_sum and cur_max
    cur_sum = vfmv_v_f_f32m1(0, vl);
    cur_max = vfmv_v_f_f32m1(-999999, vl);

    for (int j = 0; j < col; j++) {
      // x
      vfloat32m1_t vec = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);

      // m[j] = max(m[j-1], x)
      next_max = vfmax_vv_f32m1(cur_max, vec, vl);
      // vec = exp(x - m[j])
      vec = vfsub_vv_f32m1(vec, next_max, vl);
      vec = __exp_2xf32(vec, vl);
      // vec = vfmv_v_f_f32m1(5, vl);

      // tmp_vec = exp(m[j-1] - m[j])
      tmp_vec = vfsub_vv_f32m1(cur_max, next_max, vl);
      tmp_vec = __exp_2xf32(tmp_vec, vl);

      // sum[j] = exp(x - m[j]) + sum[j-1] * exp(m[j] - m[j-1]) 
      next_sum = vfmul_vv_f32m1(cur_sum, tmp_vec, vl);

      next_sum = vfadd_vv_f32m1(next_sum, vec, vl);

      cur_sum = next_sum;
      cur_max = next_max;
    }

    // =================================================
    // apply to the elements
    // =================================================

    for (int j = 0; j < col; j++) {
      vfloat32m1_t vec = vlse32_v_f32m1(&mat[i * col + j], col * 4, vl);

      // exp(x - max)
      vec = vfsub_vv_f32m1(vec, cur_max, vl);
      vec = __exp_2xf32(vec, vl);

      // exp(x - max) / sum
      vec = vfdiv_vv_f32m1(vec, cur_sum, vl);

      vsse32_v_f32m1(&mat[i * col + j], col * 4, vec, vl);
      // vsse32_v_f32m1(&mat[i * col + j], col * 4, cur_sum, vl);
    }

    i += vl;
  }
}

void softmax_transposed(float *mat, int row, int col) {
  size_t vlmax = vsetvlmax_e32m1();
  vfloat32m1_t cur_max, next_max, cur_sum, next_sum;
  vfloat32m1_t tmp_vec;

  for (int i = 0; i < col;) {
    int vl = vlmax;
    if (i + vlmax > col) vl = vsetvl_e32m1(col- i);

    // =================================================
    // update max and sum
    // =================================================

    // initialize cur_sum and cur_max
    cur_sum = vfmv_v_f_f32m1(0, vl);
    cur_max = vfmv_v_f_f32m1(-999999, vl);

    for (int j = 0; j < row; j++) {
      // x
      vfloat32m1_t vec = vle32_v_f32m1(&mat[j * col + i], vl);

      // m[j] = max(m[j-1], x)
      next_max = vfmax_vv_f32m1(cur_max, vec, vl);
      // vec = exp(x - m[j])
      vec = vfsub_vv_f32m1(vec, next_max, vl);
      vec = __exp_2xf32(vec, vl);
      // vec = vfmv_v_f_f32m1(5, vl);

      // tmp_vec = exp(m[j-1] - m[j])
      tmp_vec = vfsub_vv_f32m1(cur_max, next_max, vl);
      tmp_vec = __exp_2xf32(tmp_vec, vl);

      // sum[j] = exp(x - m[j]) + sum[j-1] * exp(m[j] - m[j-1]) 
      next_sum = vfmul_vv_f32m1(cur_sum, tmp_vec, vl);

      next_sum = vfadd_vv_f32m1(next_sum, vec, vl);

      cur_sum = next_sum;
      cur_max = next_max;
    }

    // =================================================
    // apply to the elements
    // =================================================

    for (int j = 0; j < row; j++) {
      vfloat32m1_t vec = vle32_v_f32m1(&mat[j * col + i], vl);

      // exp(x - max)
      vec = vfsub_vv_f32m1(vec, cur_max, vl);
      vec = __exp_2xf32(vec, vl);

      // exp(x - max) / sum
      vec = vfdiv_vv_f32m1(vec, cur_sum, vl);

      vse32_v_f32m1(&mat[j * col + i], vec, vl);
    }

    i += vl;
  }
}

