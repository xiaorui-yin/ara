// The APACHE License (APACHE)
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

#ifndef COMMON_H
#define COMMON_H

#include "riscv_vector.h"
#include <stdio.h>
#ifndef SPIKE
#include "printf.h"
#endif


void print_matrix(float const *matrix, int num_rows,
                  int num_columns) {
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_columns; ++j) {
      printf("%.5f ", matrix[i * num_columns + j]);
    }
    printf("  ");
  }
  printf("\n");
}

void compare_matrix(float *mat_a, float *mat_b, int row, int col) {
  int pass = 1;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      int idx = i * col + j;

      float diff;
      if (mat_b[idx] == 0) diff = mat_a[idx];
      else diff = (mat_a[idx] - mat_b[idx]) / mat_b[idx];

      if (diff > 0.01 || diff < -0.01) {
        // printf("ERROR!!! pos:(%d, %d) exp: %d, act: %d:\n", 
        //        i, j, (int)mat_b[i * col + j], (int)mat_a[i * col + j]);
        printf("ERROR!!! pos:(%d, %d) exp: %.5f, act: %.5f:\n", 
               i, j, mat_b[i * col + j], mat_a[i * col + j]);
        pass = 0;
      }
    }
  }

  if (pass == 1) printf("PASSED\n");
  else if (pass == 0) printf("FAILED\n");
}

#endif
