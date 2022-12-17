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

void relu(float *mat, int row, int col) {

#ifdef VCD_DUMP
  event_trigger = +1
#endif

  size_t vlmax = vsetvlmax_e32m8();
  long int len = row * col;

  for (int i = 0; i < len;) {
    int vl = vlmax;

    if (i + vlmax > len)
      vl = vsetvl_e32m8(len - i);

    vfloat32m8_t vec = vle32_v_f32m8(&mat[i], vl);
    vec = vfmax_vf_f32m8(vec, 0, vl);
    vse32_v_f32m8(&mat[i], vec, vl);

    i += vl;
  }

#ifdef VCD_DUMP
  event_trigger = -1
#endif
}
