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

#include "dropout.h"
#include "riscv_vector.h"

// copied from original dropout
void dropout_vec(const unsigned int n, const float *i, const float scale,
                 const int32_t *sel, float *o) {
  unsigned int vl;

  vfloat32m8_t vi, vo;
  vint32m8_t vsel;
  vbool4_t vsel_m;

  for (unsigned int avl = n; (vl = vsetvl_e32m8(avl)) > 0; avl -= vl) {
    // Load selection vector
    vsel = vle32_v_i32m8(sel, vl);
    // Produce the selection mask
    vsel_m = vmseq_vx_i32m8_b4(vsel, 1, vl);
    // Initialize output vector with zeroes
    vo = vfmv_v_f_f32m8((float)0, vl);
    // Load input vector
    vi = vle32_v_f32m8(i, vl);
    // Calculate output vector
    vo = vfmul_vf_f32m8_m(vsel_m, vo, vi, scale, vl);
    vse32_v_f32m8(o, vo, vl);
    // Bump pointers
    i += vl;
    sel += vl;
    o += vl;
  }
}

void dropout(float *mat, int *sel, float scale, const int row, const int col) {
  for (int i = 0; i < row; i++) {
    dropout_vec(col, (mat + i * col), scale, (sel + i * col), (mat + i * col));
  }
}
