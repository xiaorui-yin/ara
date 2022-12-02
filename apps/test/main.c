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
  printf("\n");

  float b[32], c[32*4];
  float a[4];
  for (int i = 0; i < 32; i++) b[i] = 1.0*i;
  for (int i = 0; i < 4; i++) a[i] = 1.0;

  int tmp = 0;
  asm volatile("vsetbl t0, %0, %1" ::"r"(32), "r"(tmp));

  asm volatile("vle32bc.v v31, (%0)" ::"r"(b));
  
  float t0 = 0;
  
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(4));
  asm volatile("vle32.v v0, (%0)" ::"r"(a));
  asm volatile("vfbmacc.vf v8, %0, v0" ::"f"(t0));

  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(32));
  asm volatile("vsse32.v v8, (%0), %1" ::"r"(c), "r"(32));

  for (int i = 0; i < 32; i++) printf("%d\n", c[i]);
  return 0;
}
