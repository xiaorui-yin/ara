// Copyright 2022 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Matteo Perotti <mperotti@iis.ee.ethz.ch>

#include <math.h>
#include <string.h>

#include "riscv_vector.h"

#include "../softmax/lib/exp.h"

// Our fdiv cannot receive any X in input
// The following macro is just a trick and should NOT be used
#define RESET_VREGS

void softmax(const float *i, const float *o, uint64_t row, uint64_t col) {

#ifdef VCD_DUMP
  event_trigger = +1
#endif

  /* ONLY FOR DEBUGGING PURPOSE. DELETE THE FOLLOWING ASM LINES
   */
  // Clean the regs from Xes
#ifdef RESET_VREGS
  volatile int temp;
  asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(temp));

  asm volatile("vmv.v.i  v0, 0");
  asm volatile("vmv.v.i  v8, 0");
  asm volatile("vmv.v.i v16, 0");
  asm volatile("vmv.v.i v24, 0");
#endif

  size_t avl = row;
  size_t vl;

  // Stripmining pointers
  float *_i = (float *)i;
  float *_o = (float *)o;
  // Channel pointers
  float *__i = (float *)i;
  float *__o = (float *)o;

  // Vector registers
  vfloat32m1_t max_chunk_v;
  vfloat32m1_t buf_chunk_v;
  vfloat32m1_t num_chunk_v;
  vfloat32m1_t den_chunk_v;
  vfloat32m1_t res_chunk_v;

  // Stripmine on col
  for (vl = vsetvl_e32m1(avl); avl > 0; avl -= vl) {

    vl = vsetvl_e32m1(avl);

    /*
      Calculate the maximum along the channel dimension
    */

    // Initialize the max vector
    max_chunk_v = vlse32_v_f32m1(__i, col * 4, vl);
    // Bump the pointer
    __i += 1;
    for (uint64_t ch = 1; ch < col; ++ch) {
      // Load a chunk of the input vector
      buf_chunk_v = vlse32_v_f32m1(__i, col * 4, vl);
      // Bump the channel pointer
      __i += 1;
      // Calculate the elm-wise maximum between the two chunks
      max_chunk_v = vfmax_vv_f32m1(max_chunk_v, buf_chunk_v, vl);
    }
    // Restore the channel pointer
    __i = _i;

    /*
      Fetch, subtract, exponentiate along the channel dimension
    */

    // Initialize accumulator
    den_chunk_v = vfmv_v_f_f32m1(0, vl);
    for (uint64_t ch = 0; ch < col; ++ch) {
      // Fetch one chunk from channel ch
      buf_chunk_v = vlse32_v_f32m1(__i, col * 4, vl);
      // Subtract the maximum
      buf_chunk_v = vfsub_vv_f32m1(buf_chunk_v, max_chunk_v, vl);
      // Exponentiate
      buf_chunk_v = __exp_2xf32(buf_chunk_v, vl);
      // Store the numerator to memory
      vsse32_v_f32m1(__o, col * 4, buf_chunk_v, vl);
      // Accumulate
      den_chunk_v = vfadd_vv_f32m1(den_chunk_v, buf_chunk_v, vl);
      // Bump channel pointers
      __i += 1;
      __o += 1;
    }
    // Restore the pointers
    __i = _i;
    __o = _o;

    /*
      Divide by the computed sum
    */

    for (uint64_t ch = 0; ch < col; ++ch) {
      // Load numerator from memory
      num_chunk_v = vlse32_v_f32m1(__o, col * 4, vl);
      // Divide
      res_chunk_v = vfdiv_vv_f32m1(num_chunk_v, den_chunk_v, vl);
      // Store the result to memory
      vsse32_v_f32m1(__o, col * 4, res_chunk_v, vl);
      // Bump channel pointers
      __o++;
    }
    // Bump stripmining pointers
    _i += vl * col;
    _o += vl * col;
    // Reset channel pointers
    __i = _i;
    __o = _o;
  }

#ifdef VCD_DUMP
  event_trigger = -1
#endif
}

void softmax_t(const float *i, const float *o, uint64_t row, uint64_t col) {

#ifdef VCD_DUMP
  event_trigger = +1
#endif

  /* ONLY FOR DEBUGGING PURPOSE. DELETE THE FOLLOWING ASM LINES
   */
  // Clean the regs from Xes
#ifdef RESET_VREGS
  volatile int temp;
  asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(temp));

  asm volatile("vmv.v.i  v0, 0");
  asm volatile("vmv.v.i  v8, 0");
  asm volatile("vmv.v.i v16, 0");
  asm volatile("vmv.v.i v24, 0");
#endif

  size_t avl = col;
  size_t vl;

  // Stripmining pointers
  float *_i = (float *)i;
  float *_o = (float *)o;
  // Channel pointers
  float *__i = (float *)i;
  float *__o = (float *)o;

  // Vector registers
  vfloat32m1_t max_chunk_v;
  vfloat32m1_t buf_chunk_v;
  vfloat32m1_t num_chunk_v;
  vfloat32m1_t den_chunk_v;
  vfloat32m1_t res_chunk_v;

  // Stripmine on col
  for (vl = vsetvl_e32m1(avl); avl > 0; avl -= vl) {

    vl = vsetvl_e32m1(avl);

    /*
      Calculate the maximum along the channel dimension
    */

    // Initialize the max vector
    max_chunk_v = vle32_v_f32m1(__i, vl);
    // Bump the pointer
    __i += col;
    for (uint64_t ch = 1; ch < row; ++ch) {
      // Load a chunk of the input vector
      buf_chunk_v = vle32_v_f32m1(__i, vl);
      // Bump the channel pointer
      __i += col;
      // Calculate the elm-wise maximum between the two chunks
      max_chunk_v = vfmax_vv_f32m1(max_chunk_v, buf_chunk_v, vl);
    }
    // Restore the channel pointer
    __i = _i;

    /*
      Fetch, subtract, exponentiate along the channel dimension
    */

    // Initialize accumulator
    den_chunk_v = vfmv_v_f_f32m1(0, vl);
    for (uint64_t ch = 0; ch < row; ++ch) {
      // Fetch one chunk from channel ch
      buf_chunk_v = vle32_v_f32m1(__i, vl);
      // Subtract the maximum
      buf_chunk_v = vfsub_vv_f32m1(buf_chunk_v, max_chunk_v, vl);
      // Exponentiate
      buf_chunk_v = __exp_2xf32(buf_chunk_v, vl);
      // Store the numerator to memory
      vse32_v_f32m1(__o, buf_chunk_v, vl);
      // Accumulate
      den_chunk_v = vfadd_vv_f32m1(den_chunk_v, buf_chunk_v, vl);
      // Bump channel pointers
      __i += col;
      __o += col;
    }
    // Restore the pointers
    __i = _i;
    __o = _o;

    /*
      Divide by the computed sum
    */

    for (uint64_t ch = 0; ch < row; ++ch) {
      // Load numerator from memory
      num_chunk_v = vle32_v_f32m1(__o, vl);
      // Divide
      res_chunk_v = vfdiv_vv_f32m1(num_chunk_v, den_chunk_v, vl);
      // Store the result to memory
      vse32_v_f32m1(__o, res_chunk_v, vl);
      // Bump channel pointers
      __o += col;
    }
    // Bump stripmining pointers
    _i += vl;
    _o += vl;
    // Reset channel pointers
    __i = _i;
    __o = _o;
  }

#ifdef VCD_DUMP
  event_trigger = +1
#endif
}
