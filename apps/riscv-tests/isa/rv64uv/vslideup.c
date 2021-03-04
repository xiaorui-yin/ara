// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
//         Basile Bougenot <bbougenot@student.ethz.ch>

#include "vector_macros.h"

void TEST_CASE1() {
  VSET(6,e32,m1);
  VLOAD_32(v2,1,2,3,0,0,1);
  __asm__ volatile("vslideup.vi v1, v2, 3");
  VEC_CMP_32(1,v1,0,0,0,1,2,3);
}

void TEST_CASE2() {
  VSET(6,e32,m1);
  VLOAD_32(v2,1,2,3,0,0,1);
  VLOAD_U32(v0,15,0,0,0,0,0);
  CLEAR(v2);
  __asm__ volatile("vslideup.vi v1, v2, 3, v0.t");
  VEC_CMP_32(2,v1,0,0,0,0,2,3);
}

void TEST_CASE3() {
  VSET(6,e32,m1);
  VLOAD_32(v2,1,2,3,0,0,1);
  uint64_t scalar = 3;
  __asm__ volatile("vslideup.vx v1, v2, %[A]":: [A] "r" (scalar));
  VEC_CMP_32(3,v1,0,0,0,1,2,3);
}

void TEST_CASE4() {
  VSET(6,e32,m1);
  VLOAD_32(v2,1,2,3,0,0,1);
  uint64_t scalar = 3;
  VLOAD_U32(v0,15,0,0,0,0,0);
  CLEAR(v2);
  __asm__ volatile("vslideup.vx v1, v2, %[A], v0.t":: [A] "r" (scalar));
  VEC_CMP_32(4,v1,0,0,0,0,2,3);
}


int main(void){
  INIT_CHECK();
  enable_vec();
  TEST_CASE1();
  TEST_CASE2();
  TEST_CASE3();
  TEST_CASE4();
  EXIT_CHECK();
}