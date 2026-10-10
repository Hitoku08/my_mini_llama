// SPDX-License-Identifier: MIT

#ifndef TESTS_TEST_NAMES_H_
#define TESTS_TEST_NAMES_H_

// 由 scripts/step.py 按步骤追加，只包含已完成模块的头文件和符号。
// 完整列表见原项目 tests/test_names.h。

#include "mini_llama/tensor.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "mini_llama/tokenizer.h"
// @includes

using mini_llama::MakeTensor1D;
using mini_llama::MakeTensor2D;
using mini_llama::MakeTensor3D;
using mini_llama::MakeTensor4D;
using mini_llama::Tensor;
using mini_llama::RopeType;
using mini_llama::ArgMax;
using mini_llama::Linear;
using mini_llama::Matmul;
using mini_llama::RmsNorm;
using mini_llama::Rope;
using mini_llama::Softmax;
using mini_llama::SwiGlu;
using mini_llama::AsciiTokenizer;
using mini_llama::CreateTokenizer;
using mini_llama::ITokenizer;
using mini_llama::JsonVocabTokenizer;
// @usings

#endif  // TESTS_TEST_NAMES_H_
