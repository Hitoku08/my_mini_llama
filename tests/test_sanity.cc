// SPDX-License-Identifier: MIT

#include "tests/test_main.h"

static bool TestSanityOnePlusOne() {
  MINI_LLAMA_ASSERT_EQ(1 + 1, 2);
  return true;
}

static struct SanityTestRegistrar {
  SanityTestRegistrar() {
    RegisterTest("sanity_one_plus_one", TestSanityOnePlusOne);
  }
} sanity_test_registrar;
