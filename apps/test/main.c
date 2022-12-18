#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef SPIKE
#include "printf.h"
#endif

int main() {
  printf("\n");
  printf("=============\n");
  printf("=HELLO WORLD=\n");
  printf("=============\n");
  printf("\n");

  float a[16] __attribute__((aligned(32 * NR_LANES)));
  float b[32] __attribute__((aligned(32 * NR_LANES)));
  float a_[16] __attribute__((aligned(32 * NR_LANES)));
  float b_[32] __attribute__((aligned(32 * NR_LANES)));
  float c1[128] __attribute__((aligned(32 * NR_LANES)));
  float c2[128] __attribute__((aligned(32 * NR_LANES)));
  float c3[128] __attribute__((aligned(32 * NR_LANES)));
  float c4[128] __attribute__((aligned(32 * NR_LANES)));

  for (int i = 0; i < 32; i++) {
    b[i] = 1.0*i;
    b_[i] = 2.0*i;
  }
  for (int i = 0; i < 16; i++) {
    a[i] = 1.0;
    a_[i] = 2.0;
  }

  for (int i = 0; i < 128; i++) {
    c1[i] = 5;
    c2[i] = 5;
    c3[i] = 5;
    c4[i] = 5;
  }

  int tmp = 0;
  // Set vtype, vl is meanlingless in this case
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(4));

  asm volatile("vsetbl t0, %0, %1" ::"r"(32), "r"(tmp));

  asm volatile("vle32bc.v v31, (%0)" ::"r"(b));

  float t0 = 0;

  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(16));
  asm volatile("vle32.v v0, (%0)" ::"r"(a));
  asm volatile("vfbmacc.vf v8, %0, v0" ::"f"(t0));

  asm volatile("vle32bc.v v31, (%0)" ::"r"(b_));
  asm volatile("vle32.v v0, (%0)" ::"r"(a_));

  asm volatile("vfbmacc.vv v8, v31, v0");

  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(32*4)); // NrLanes * bl
  asm volatile("vsse32.v v8, (%0), %1" ::"r"(c1), "r"(4*4)); // row4, row5, row6, row7
  asm volatile("vsse32.v v9, (%0), %1" ::"r"(c2), "r"(4*4)); // row4, row5, row6, row7
  asm volatile("vsse32.v v10, (%0), %1" ::"r"(c3), "r"(4*4)); // row8, row9, row10, row11
  asm volatile("vsse32.v v11, (%0), %1" ::"r"(c4), "r"(4*4)); // row12, row13, row14, row15

  printf("Row 1-4\n");
  for (int i = 0; i < 128; i++) {
    // printf("ADDR: %p, DATA:%f\n", &c1[i], c1[i]);
    printf("DATA:%d\n", (int)c1[i]);
  }
  printf("Row 5-8\n");
  for (int i = 0; i < 128; i++) {
    // printf("ADDR: %p, DATA:%f\n", &c2[i], c2[i]);
    printf("DATA:%d\n", (int)c2[i]);
  }
  printf("Row 9-12\n");
  for (int i = 0; i < 128; i++) {
    // printf("ADDR: %p, DATA:%f\n", &c3[i], c3[i]);
    printf("DATA:%d\n", (int)c3[i]);
  }
  printf("Row 13-16\n");
  for (int i = 0; i < 128; i++) {
    // printf("ADDR: %p, DATA:%f\n", &c4[i], c4[i]);
    printf("DATA:%d\n", (int)c4[i]);
  }
  return 0;
}
