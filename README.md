# my-mini-llama

从零手写一个 LLaMA-style 的 C++ 推理引擎，目标是完全理解 [mini-llama-homework](../mini-llama-homework) 这个参考项目。

参考项目只读，本项目里的代码全部自己写。学习路线分 6 个阶段、30 个小步，每完成一步都能用测试或命令行验收。

## 环境

- C++17 编译器、CMake 3.17+
- Python 3 和 numpy（Python 参考实现、golden logits 对齐会用到）
- 参考项目放在 `../mini-llama-homework`；放在别处时用环境变量 `MINI_LLAMA_REF` 指定路径
- CUDA 部分可选，需要 NVIDIA GPU 和 CUDA toolkit

## 快速开始

```bash
python3 scripts/step.py list       # 查看 30 步和当前进度
python3 scripts/step.py start 1    # 进入第 1 步
# 读参考项目的头文件，自己写 include/mini_llama/tensor.h 和 src/tensor.cc
python3 scripts/step.py check      # 编译并运行全部测试
python3 scripts/step.py done 1     # check 通过后标记完成
```

## step.py 做了什么

| 命令 | 作用 |
|---|---|
| `list` | 列出所有步骤，`[x]` 已完成，`[~]` 进行中 |
| `show N` | 打印第 N 步的目标、要读的文件、要写的文件、验收方式和注意事项 |
| `start N` | 拷贝这一步的原测试和脚本，更新 `CMakeLists.txt` 和 `tests/test_names.h`，生成笔记模板 |
| `check` | `cmake` 配置、编译、`ctest` |
| `done N` | 先跑 `check`，通过后标记第 N 步完成 |

`start` 只会为源文件创建空的 `.cc` 占位，头文件和实现都由你自己写。进度保存在 `.step_state.json`。

`CMakeLists.txt` 和 `tests/test_names.h` 里以 `# @xxx`、`// @xxx` 开头的行是脚本的插入点，不要删除。

## 学习路线

| 阶段 | 步骤 | 内容 | 里程碑 |
|---|---|---|---|
| 0 | - | 脚手架、测试框架 | `ctest` 通过 1 个测试（已完成） |
| 1 | 1-4 | Tensor、基础算子、基础 tokenizer、JSON+BIN loader、`inspect` | 单元测试通过 |
| 2 | 5-11 | KV cache、context、batch、forward、sampler、`generate`、golden logits、chat、`run` | tiny 模型可生成，与 Python 参考实现对齐 |
| 3 | 12-14 | benchmark、Q8_0、Q4_0/Q4_1 量化 | `bench` 可用，量化路径误差在阈值内 |
| 4 | 15-21 | GGUF reader、GGUF loader、BPE、GGUF tokenizer、Qwen2 chat template | `run models/chat` 能真实对话 |
| 5 | 22-25 | matmul dispatch、线程池、AVX2 SIMD、Python 对齐进阶 | 参考项目所有非 CUDA 测试通过 |
| 6 | 26-30 | backend 选择、CUDA 构建开关、CUDA 接口与 kernel | CPU 构建全绿；有 GPU 时 CUDA 测试全绿 |

量化（阶段 3）排在真实模型（阶段 4）之前，因为内置的 Qwen2 模型是 Q8_0 格式，加载器需要 `QuantizedTensor`。

## 跨多步的测试文件

部分原测试文件覆盖了多个步骤的功能。进入较早的那一步时，先把依赖后续功能的用例注释掉，并标上 `// TODO(stepN)`，到第 N 步再恢复：

- `test_ops.cc`：量化相关用例，第 13 步恢复
- `test_tokenizer.cc`：BPE 用例，第 18 步恢复
- `test_quant.cc`：Q4_0 / Q4_1 用例，第 14 步恢复
- `test_chat.cc`、`test_gguf_tokenizer.cc`：chat template 用例，第 20 步恢复
- `test_threaded_matmul.cc`：SIMD 用例，第 24 步恢复

每一步 `start` 时打印的“注意”里也会提示这些内容。

## 目录结构

```text
my-mini-llama/
  CMakeLists.txt        # 随步骤自动扩展
  include/mini_llama/   # 你写的头文件
  src/                  # 你写的实现和 main.cc
  tests/                # 测试框架 + 从参考项目拷贝的测试
  scripts/step.py       # 步骤管理工具；其他脚本随步骤拷贝进来
  models/tiny/          # tiny 模型、vocab、测试用 GGUF
  models/chat/          # Qwen2-0.5B-Instruct-Q8_0.gguf（第 19 步再导出 tokenizer 文件）
  notes/                # 每一步的笔记和思考题
```

真实模型约 500MB，已加入 `.gitignore`。缺失时运行 `python3 scripts/download_demo_model.py` 下载（第 17 步会拷贝这个脚本）。

## 常用命令

```bash
# 手动构建和测试
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure

# 只看某个 Python 回归测试
ctest --test-dir build -R golden-logits --output-on-failure

# CUDA 构建（第 27 步之后，需要 NVIDIA GPU）
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

## 建议

- 每一步先读参考项目的头文件，自己写；卡住超过 30 分钟再看参考实现的 `src/*.cc`。
- 用 `git` 管理本项目，每完成一步提交一次，之后可以用 `git diff` 回看自己写了什么。
- 每一步都填写 `notes/` 里的思考题，重点记录数据流、tensor shape 和边界条件。
