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

#ifndef SELF_ATTENTION_H
#define SELF_ATTENTION_H

void self_attention(float *o, float *q, float *k, float *v, int n, int d_model,
                    int dk);

void self_attention_t(float *o, float *q, float *k, float *v, int n,
                      int d_model, int dk);

#endif
