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
#ifndef SPIKE
#include "printf.h"
#endif

// TODO: V2, store mask to the memory
void relu(float *mat, int row, int col) {
  size_t vlmax = vsetvlmax_e32m1();

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col;) {
      int vl = vlmax;
      // if (j + vlmax > col) vl = col - j;
      if (j + vlmax > col)
        vl = vsetvl_e32m1(col - j);

      vfloat32m1_t vec = vle32_v_f32m1(&mat[i * col + j], vl);
      vec = vfmax_vf_f32m1(vec, 0, vl);
      vse32_v_f32m1(&mat[i * col + j], vec, vl);

      j += vl;
    }
  }
}
