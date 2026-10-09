cmake_minimum_required(VERSION 3.17)
project(mini-llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 以 "# @xxx" 开头的行是 scripts/step.py 的插入点，请勿删除。
# @options

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

set(MINI_LLAMA_CORE_SOURCES
    src/tensor.cc
    # @sources
)
# @cuda_sources

set(TEST_SOURCES
    tests/test_main.cc
    tests/test_sanity.cc
    tests/test_tensor.cc
    # @tests
)

if(MINI_LLAMA_CORE_SOURCES)
    add_library(mini-llama-core STATIC ${MINI_LLAMA_CORE_SOURCES})
endif()
# @core_link

add_executable(mini-llama-tests ${TEST_SOURCES})
if(TARGET mini-llama-core)
    target_link_libraries(mini-llama-tests PRIVATE mini-llama-core)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/src/main.cc)
    add_executable(mini-llama src/main.cc)
    target_link_libraries(mini-llama PRIVATE mini-llama-core)
endif()

enable_testing()
add_test(NAME mini-llama-tests COMMAND mini-llama-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# @extra

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    # @pytests
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
