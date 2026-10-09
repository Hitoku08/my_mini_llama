// SPDX-License-Identifier: MIT

#ifndef TESTS_TEST_NAMES_H_
#define TESTS_TEST_NAMES_H_

// 由 scripts/step.py 按步骤追加，只包含已完成模块的头文件和符号。
// 完整列表见原项目 tests/test_names.h。

#include "mini_llama/tensor.h"
// @includes

using mini_llama::MakeTensor1D;
using mini_llama::MakeTensor2D;
using mini_llama::MakeTensor3D;
using mini_llama::MakeTensor4D;
using mini_llama::Tensor;
// @usings

#endif  // TESTS_TEST_NAMES_H_
