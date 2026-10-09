#!/usr/bin/env python3
"""mini-llama 重写学习路线的步骤管理工具。

用法:
  python3 scripts/step.py list            查看所有步骤和进度
  python3 scripts/step.py show <N>        查看第 N 步的任务、参考文件和验收方式
  python3 scripts/step.py start <N>       进入第 N 步: 拷贝测试/脚本, 更新 CMake 和 test_names.h
  python3 scripts/step.py check           编译并运行全部测试
  python3 scripts/step.py done <N>        check 通过后把第 N 步标记为完成

原项目路径默认是 ../mini-llama-homework, 可用环境变量 MINI_LLAMA_REF 覆盖。
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("MINI_LLAMA_REF", ROOT.parent / "mini-llama-homework")).resolve()
STATE_FILE = ROOT / ".step_state.json"
CMAKE_FILE = ROOT / "CMakeLists.txt"
NAMES_FILE = ROOT / "tests" / "test_names.h"
CHAT_GGUF = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf"


def pytest_block(name, script):
    return (
        f"add_test(NAME {name} COMMAND ${{CMAKE_COMMAND}} -E env "
        f"MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama> ${{Python3_EXECUTABLE}} "
        f"${{CMAKE_SOURCE_DIR}}/scripts/{script} WORKING_DIRECTORY ${{CMAKE_SOURCE_DIR}})"
    )


def cuda_test_exe(exe, src, tests):
    lines = [
        f"add_executable({exe} {src})",
        f"target_link_libraries({exe} PRIVATE mini-llama-core)",
    ]
    for name, arg in tests:
        cmd = f"{exe} {arg}".strip()
        lines.append(
            f"add_test(NAME {name} COMMAND {cmd} WORKING_DIRECTORY ${{CMAKE_SOURCE_DIR}})"
        )
    return lines


CUDA_OPTIONS = [
    'option(MINI_LLAMA_CUDA "Build NVIDIA CUDA backend" OFF)',
    "if(MINI_LLAMA_CUDA)",
    "    enable_language(CUDA)",
    "    if(NOT CMAKE_CUDA_ARCHITECTURES)",
    "        set(CMAKE_CUDA_ARCHITECTURES 89)",
    "    endif()",
    "    find_package(CUDAToolkit REQUIRED)",
    "    add_compile_definitions(MINI_LLAMA_USE_CUDA=1)",
    "endif()",
]

CUDA_CORE_LINK = [
    "if(MINI_LLAMA_CUDA)",
    "    target_link_libraries(mini-llama-core PUBLIC CUDA::cudart CUDA::cublas)",
    "endif()",
]

# 每一步的字段:
#   stage / title / goal     阶段、标题、目标
#   read                     建议阅读的原项目文件（卡住时再看 src 实现）
#   write                    你需要自己写的文件
#   sources                  加入 MINI_LLAMA_CORE_SOURCES 的 .cc（不存在时创建空文件）
#   includes / usings        追加到 tests/test_names.h
#   tests                    从原项目拷贝并加入 mini-llama-tests 的测试
#   copy                     只拷贝不编译的文件（fixture、脚本等）
#   pytests                  (ctest 名称, 脚本名) 形式的 Python 回归测试
#   cmake                    {插入点: [行...]} 形式的额外 CMake 片段
#   verify                   手动验收命令
#   notes                    注意事项
#   questions                写进 notes/ 的思考题
STEPS = {
    1: dict(
        stage=1, title="Tensor", goal="理解张量的 shape、连续存储和边界检查",
        read=["include/mini_llama/tensor.h", "src/tensor.cc"],
        write=["include/mini_llama/tensor.h", "src/tensor.cc"],
        sources=["src/tensor.cc"],
        includes=["tensor.h"],
        usings=["MakeTensor1D", "MakeTensor2D", "MakeTensor3D", "MakeTensor4D", "Tensor"],
        tests=["test_tensor.cc"],
        questions=["shape {2,3,4} 的张量，元素 (i,j,k) 在 data 里的下标是多少？",
                   "为什么 at2 / row_ptr 要做越界检查，而不是直接用 data[]？"],
    ),
    2: dict(
        stage=1, title="基础算子 ops", goal="手写 matmul、rmsnorm、softmax、silu、swiglu、RoPE",
        read=["include/mini_llama/ops.h", "src/ops.cc", "include/mini_llama/model.h"],
        write=["include/mini_llama/ops.h", "src/ops.cc", "include/mini_llama/model.h (先只放 RopeType)"],
        sources=["src/ops.cc"],
        includes=["model.h", "ops.h"],
        usings=["RopeType", "ArgMax", "Linear", "Matmul", "RmsNorm", "Rope", "Softmax", "SwiGlu"],
        tests=["test_ops.cc"],
        notes=["原 ops.h 依赖 model.h（RopeType）和 quantized_tensor.h。这一步的 model.h 只需要定义 RopeType，"
               "量化相关的重载留到第 13 步。",
               "test_ops.cc 里如果有用到 QuantizedTensor 的用例，先注释掉并标 // TODO(step13)。"],
        questions=["softmax 为什么要先减去最大值？", "rmsnorm 和 layernorm 的区别是什么？",
                   "normal RoPE 和 NeoX RoPE 配对旋转的维度有什么不同？"],
    ),
    3: dict(
        stage=1, title="基础 Tokenizer", goal="实现 ASCII tokenizer 和 JSON vocab tokenizer",
        read=["include/mini_llama/tokenizer.h", "src/tokenizer.cc", "src/json_vocab_tokenizer.cc"],
        write=["include/mini_llama/tokenizer.h", "src/tokenizer.cc", "src/json_vocab_tokenizer.cc"],
        sources=["src/tokenizer.cc", "src/json_vocab_tokenizer.cc"],
        includes=["tokenizer.h"],
        usings=["AsciiTokenizer", "CreateTokenizer", "ITokenizer", "JsonVocabTokenizer"],
        tests=["test_tokenizer.cc"],
        notes=["BPE 相关用例（BpeTokenizer / CreateBpeTokenizer）先注释掉并标 // TODO(step18)。"],
        questions=["ITokenizer 接口为什么要把 encode / decode / decode_token 分开？",
                   "vocab 里找不到的字符应该怎么处理？"],
    ),
    4: dict(
        stage=1, title="Model + JSON/BIN Loader + inspect", goal="理解模型配置、权重布局和加载校验",
        read=["include/mini_llama/model.h", "include/mini_llama/loader.h", "src/loader.cc",
              "src/main.cc (只看 inspect 部分)", "models/tiny/model.json"],
        write=["include/mini_llama/model.h (F32 版)", "include/mini_llama/loader.h", "src/loader.cc",
               "src/main.cc (只支持 inspect)"],
        sources=["src/loader.cc"],
        includes=["loader.h"],
        usings=["LayerWeights", "MiniLlamaModel", "ModelConfig", "LoadModel", "ModelManifest",
                "ParseManifest"],
        tests=["test_loader.cc"],
        verify=["./build/mini-llama inspect models/tiny"],
        notes=["model.h 这一版不要引入 quantized_tensor.h / cuda_runtime.h。",
               "写了 src/main.cc 之后需要重新跑 cmake，mini-llama 可执行文件才会生成。"],
        questions=["manifest 里的 offset / size 为什么要检查重叠和尾部多余字节？",
                   "画出 tiny 模型每层有哪些权重，以及各自的 shape。"],
    ),
    5: dict(
        stage=2, title="KV Cache + Context", goal="理解 KV Cache 的设计以及 prefill / decode 的状态变化",
        read=["include/mini_llama/kv_cache.h", "src/kv_cache.cc", "include/mini_llama/context.h",
              "src/context.cc"],
        write=["include/mini_llama/kv_cache.h", "src/kv_cache.cc", "include/mini_llama/context.h",
               "src/context.cc"],
        sources=["src/kv_cache.cc", "src/context.cc"],
        includes=["kv_cache.h", "context.h"],
        usings=["KvCache", "MiniLlamaContext"],
        tests=["test_kv_cache.cc"],
        notes=["context.h 先不要包含 cuda_kv_cache.h，CUDA 字段留到第 28 步。"],
        questions=["画出 KV cache 的 shape: [n_layer][max_seq][n_kv_head * head_dim]。",
                   "prefill 结束和每次 decode 之后，context 里哪些字段会变化？"],
    ),
    6: dict(
        stage=2, title="MiniBatch", goal="统一 prefill 和 decode 的输入表示",
        read=["include/mini_llama/batch.h", "src/batch.cc"],
        write=["include/mini_llama/batch.h", "src/batch.cc"],
        sources=["src/batch.cc"],
        includes=["batch.h"],
        usings=["MiniBatch"],
        notes=["这一步没有独立的测试，能编译通过即可；test_batch.cc 在第 7 步加入。"],
        questions=["from_tokens 和 single 分别对应 llama.cpp 里的什么？"],
    ),
    7: dict(
        stage=2, title="Forward 推理主链路", goal="embed -> rmsnorm -> qkv -> rope -> attention -> ffn -> logits",
        read=["include/mini_llama/forward.h", "src/forward.cc", "scripts/reference_forward.py"],
        write=["include/mini_llama/forward.h", "src/forward.cc"],
        sources=["src/forward.cc"],
        includes=["forward.h"],
        usings=["ForwardBatch", "ForwardToken"],
        tests=["test_batch.cc"],
        copy=["scripts/reference_forward.py"],
        verify=["python3 scripts/reference_forward.py --config models/tiny/model.json "
                "--weights models/tiny/model.bin --prompt hello -n 16"],
        notes=["按子函数一个一个写: embed_token -> forward_layer(rmsnorm, qkv, rope, attention_forward, "
               "residual, ffn_forward) -> compute_logits，每写完一个就编译一次。",
               "test_forward.cc 依赖 sampler.h，放到第 8 步加入。"],
        questions=["GQA 里一个 KV head 被几个 Q head 共享？代码里怎么算下标？",
                   "attention 为什么要乘 1/sqrt(head_dim)？",
                   "为什么 prefill 一次处理多个 token 和逐个 decode 的结果应该完全相同？"],
    ),
    8: dict(
        stage=2, title="Sampler + generate 命令", goal="实现 greedy / temperature / top-k，固定 seed 可复现",
        read=["include/mini_llama/sampler.h", "src/sampler.cc", "src/main.cc (generate 部分)"],
        write=["include/mini_llama/sampler.h", "src/sampler.cc", "src/main.cc (增加 generate)"],
        sources=["src/sampler.cc"],
        includes=["sampler.h"],
        usings=["MiniSampler", "SampleGreedy", "SamplingParams"],
        tests=["test_sampler.cc", "test_forward.cc"],
        verify=["./build/mini-llama generate --config models/tiny/model.json --model models/tiny/model.bin -p hello -n 16",
                "python3 scripts/reference_forward.py --config models/tiny/model.json "
                "--weights models/tiny/model.bin --prompt hello -n 16",
                "# 两者输出的 generated tokens 行必须完全一致"],
        notes=["generate 的输出里要有一行 `generated tokens: [...]`，后面的 Python 回归测试会解析这一行。"],
        questions=["temperature -> 0 时为什么等价于 greedy？", "top-k 之后为什么要重新归一化？"],
    ),
    9: dict(
        stage=2, title="Golden logits 基础版", goal="用 Python 参考实现作为数值正确性基线",
        read=["scripts/make_golden.py", "scripts/compare_logits.py", "scripts/test_golden.py",
              "src/main.cc (--dump-logits 部分)"],
        write=["src/main.cc (增加 --dump-logits)"],
        copy=["scripts/make_golden.py", "scripts/compare_logits.py", "scripts/test_golden.py"],
        pytests=[("golden-logits", "test_golden.py")],
        notes=["--dump-logits 的文件格式要和 compare_logits.py 读取的格式一致，先读脚本再写 C++。"],
        questions=["logits 误差阈值为什么是 1e-4 而不是 0？"],
    ),
    10: dict(
        stage=2, title="RequestContext + RadixTree", goal="请求级追踪与前缀树",
        read=["include/mini_llama/request_context.h", "src/request_context.cc",
              "include/mini_llama/radix_tree.h", "src/radix_tree.cc"],
        write=["include/mini_llama/request_context.h", "src/request_context.cc",
               "include/mini_llama/radix_tree.h", "src/radix_tree.cc"],
        sources=["src/request_context.cc", "src/radix_tree.cc"],
        includes=["request_context.h"],
        usings=["FormatRequestTraceEvents", "FormatRequestTraceSummary", "RequestContext",
                "StartRequest", "TraceEvent"],
        tests=["test_request_context.cc", "test_radix_tree.cc"],
        questions=["radix tree 在多轮对话里可以用来复用什么？"],
    ),
    11: dict(
        stage=2, title="Chat / PromptBuilder / Terminal + run 模式", goal="终端聊天程序设计",
        read=["include/mini_llama/chat.h", "src/chat.cc", "include/mini_llama/prompt_builder.h",
              "src/prompt_builder.cc", "include/mini_llama/terminal.h", "src/terminal.cc",
              "src/main.cc (run 部分)"],
        write=["include/mini_llama/chat.h", "src/chat.cc", "include/mini_llama/prompt_builder.h",
               "src/prompt_builder.cc", "include/mini_llama/terminal.h", "src/terminal.cc",
               "src/main.cc (增加 run)"],
        sources=["src/chat.cc", "src/prompt_builder.cc", "src/terminal.cc"],
        includes=["chat.h", "prompt_builder.h", "terminal.h"],
        usings=["ChatMessage", "ChatSession", "PromptBuilder", "Terminal"],
        tests=["test_chat.cc"],
        verify=["printf '/help\\n/stats\\n/exit\\n' | ./build/mini-llama run models/tiny --tokenizer models/tiny/vocab.json"],
        notes=["Qwen2 chat template 相关用例先注释掉并标 // TODO(step20)。",
               "tiny 模型输出是乱码，属于正常现象。"],
        questions=["/clear 需要重置哪些状态（session、context、KV cache）？"],
    ),
    12: dict(
        stage=3, title="Debug / Benchmark", goal="建立 prefill / decode 分段性能统计",
        read=["include/mini_llama/debug.h", "src/debug.cc", "src/main.cc (bench 部分)"],
        write=["include/mini_llama/debug.h", "src/debug.cc", "src/main.cc (增加 bench)"],
        sources=["src/debug.cc"],
        includes=["debug.h"],
        usings=["BenchmarkResult", "DumpKvCacheInfo", "DumpLogitsTopK", "DumpTensorShape", "RunBenchmark"],
        tests=["test_debug.cc"],
        copy=["scripts/test_mvp.py"],
        pytests=[("mvp-regression", "test_mvp.py")],
        verify=["./build/mini-llama bench models/tiny --tokenizer models/tiny/vocab.json -p hello -n 16 --verbose"],
        notes=["test_mvp.py 会同时检查 generate、run、inspect、bench 四个命令，所以放到这一步才加入。"],
        questions=["为什么 prefill 的 tokens/s 通常比 decode 高很多？"],
    ),
    13: dict(
        stage=3, title="Q8_0 量化", goal="掌握 Q8_0 的 block、scale 和反量化 matmul",
        read=["include/mini_llama/quantized_tensor.h", "src/quantized_tensor.cc",
              "include/mini_llama/quant.h", "src/quant.cc", "src/model.cc (QuantizeModelToQ80)"],
        write=["include/mini_llama/quantized_tensor.h", "src/quantized_tensor.cc",
               "include/mini_llama/quant.h", "src/quant.cc", "src/model.cc",
               "model.h / ops.h 增加 QuantizedTensor 支持", "src/main.cc (增加 --quant q8_0)"],
        sources=["src/quantized_tensor.cc", "src/quant.cc", "src/model.cc"],
        includes=["quantized_tensor.h", "quant.h"],
        usings=["BlockQ80", "kQ80BlockSize", "QuantizedTensor", "QuantType", "ToTensor",
                "DequantizeFromQ80", "LinearQ80", "MatmulQ80", "QuantizeToQ80", "QuantizeModelToQ80"],
        tests=["test_quant.cc"],
        verify=["./build/mini-llama generate --config models/tiny/model.json --model models/tiny/model.bin "
                "-p hello -n 16 --quant q8_0"],
        notes=["test_quant.cc 里 Q4_0 / Q4_1 的用例先注释掉并标 // TODO(step14)。",
               "恢复第 2 步里标了 TODO(step13) 的 ops 用例。"],
        questions=["一个 Q8_0 block 占多少字节？相比 F32 压缩了多少？",
                   "scale 为什么用 fp16 存储？"],
    ),
    14: dict(
        stage=3, title="Q4_0 / Q4_1 量化", goal="掌握 nibble 存储与解码",
        read=["include/mini_llama/quantized_tensor.h", "src/quant.cc", "scripts/test_quant_cli.py"],
        write=["quantized_tensor.h / quant.h / quant.cc / model.cc 增加 Q4_0、Q4_1", "src/main.cc (--quant q4_0)"],
        usings=["BlockQ40", "BlockQ41", "kQ40BlockSize", "kQ41BlockSize", "DequantizeFromQ40",
                "DequantizeFromQ41", "LinearQ40", "LinearQ41", "MatmulQ40", "QuantizeToQ40",
                "QuantizeModelToQ40"],
        copy=["scripts/test_quant_cli.py"],
        pytests=[("quant-cli-smoke", "test_quant_cli.py")],
        verify=["./build/mini-llama bench models/tiny --tokenizer models/tiny/vocab.json -n 16 --quant q4_0"],
        notes=["恢复 test_quant.cc 里标了 TODO(step14) 的用例。"],
        questions=["Q4_1 比 Q4_0 多了 min 字段，换来了什么？"],
    ),
    15: dict(
        stage=4, title="GGUF 格式总览", goal="说清 header、metadata、tensor info、tensor data 四部分职责",
        read=["include/mini_llama/gguf.h", "scripts/make_test_gguf.py"],
        verify=["xxd models/tiny/test.gguf | head -20   # 对照笔记找出 magic、version、tensor_count、kv_count"],
        notes=["这一步只读不写，把四部分的二进制布局画在笔记里。"],
        questions=["GGUF 里 tensor data 为什么需要 alignment？",
                   "metadata 的 value 有哪些类型？数组类型怎么编码？"],
    ),
    16: dict(
        stage=4, title="GGUF Reader + inspect-gguf", goal="实现底层 GGUF 读取能力",
        read=["include/mini_llama/gguf.h", "src/gguf.cc", "src/main.cc (inspect-gguf 部分)"],
        write=["include/mini_llama/gguf.h", "src/gguf.cc", "src/main.cc (增加 inspect-gguf)"],
        sources=["src/gguf.cc"],
        includes=["gguf.h"],
        usings=["GgmlType", "GgmlTypeName", "GgufMetadataKv", "GgufReader", "GgufTensorInfo",
                "GgufValueType", "GgufValueTypeName", "kGgmlTypeF16", "kGgmlTypeF32", "kGgmlTypeQ40",
                "kGgmlTypeQ80", "kGgufTypeArray", "kGgufTypeString", "kGgufTypeUint32"],
        tests=["test_gguf.cc"],
        copy=["scripts/make_test_gguf.py"],
        verify=["./build/mini-llama inspect-gguf models/tiny/test.gguf"],
        questions=["读到一个未知的 metadata 类型时应该报错还是跳过？为什么？"],
    ),
    17: dict(
        stage=4, title="GGUF -> MiniLlamaModel 映射", goal="把 Qwen2 GGUF 的核心 tensor 映射到 MiniLlamaModel",
        read=["include/mini_llama/gguf_loader.h", "src/gguf_loader.cc"],
        write=["include/mini_llama/gguf_loader.h", "src/gguf_loader.cc", "src/main.cc (--model 支持 .gguf)"],
        sources=["src/gguf_loader.cc"],
        includes=["gguf_loader.h"],
        usings=["LoadGgufModel"],
        tests=["test_gguf_loader.cc"],
        copy=["tests/gguf_loader_fixture.h", "scripts/download_demo_model.py"],
        link_chat_model=True,
        verify=[f"./build/mini-llama inspect-gguf {CHAT_GGUF}",
                f"./build/mini-llama generate --model {CHAT_GGUF} -p hello -n 1"],
        notes=["真实模型约 500MB。原项目里有真实文件时会创建软链接，否则用 download_demo_model.py 从 ModelScope 下载。",
               "Qwen2 的 q/k/v projection 带 bias，tiny 模型没有，注意兼容两种情况。"],
        questions=["blk.N.attn_q.weight 在 GGUF 里的 shape 和 MiniLlamaModel 里的 shape 是否需要转置？"],
    ),
    18: dict(
        stage=4, title="BPE Tokenizer 原理", goal="理解 vocab、merges、rank、special token",
        read=["src/bpe_tokenizer.cc", "include/mini_llama/tokenizer.h"],
        write=["src/bpe_tokenizer.cc", "tokenizer.h 增加 BpeTokenizer"],
        sources=["src/bpe_tokenizer.cc"],
        usings=["BpeTokenizer", "CreateBpeTokenizer"],
        notes=["恢复 test_tokenizer.cc 里标了 TODO(step18) 的用例。"],
        questions=["用 merges [(l,o),(lo,w),(e,r)] 手推 \"lower\" 的分词过程。",
                   "byte-level BPE 为什么不会出现 unknown token？"],
    ),
    19: dict(
        stage=4, title="GGUF Tokenizer 接入", goal="接入真实 tokenizer 数据",
        read=["scripts/export_gguf_tokenizer.py", "include/mini_llama/gguf_tokenizer.h", "src/gguf_tokenizer.cc"],
        write=["include/mini_llama/gguf_tokenizer.h", "src/gguf_tokenizer.cc"],
        sources=["src/gguf_tokenizer.cc"],
        includes=["gguf_tokenizer.h"],
        usings=["CreateGgufTokenizer"],
        tests=["test_gguf_tokenizer.cc"],
        copy=["tests/gguf_tokenizer_fixture.h", "scripts/export_gguf_tokenizer.py"],
        verify=[f"python3 scripts/export_gguf_tokenizer.py {CHAT_GGUF} --out-dir models/chat"],
        notes=["先运行 verify 里的导出命令生成 vocab.json / merges.txt / special_tokens.json，再跑测试。",
               "chat template 相关用例先注释掉并标 // TODO(step20)。"],
        questions=["<|im_start|> 这类 special token 为什么不能参与普通的 BPE 合并？"],
    ),
    20: dict(
        stage=4, title="Chat Template 接入", goal="Qwen2 system/user/assistant 模板生成正确 prompt",
        read=["src/prompt_builder.cc", "include/mini_llama/prompt_builder.h"],
        write=["prompt_builder.h / prompt_builder.cc 增加 Qwen2 模板和 LoadChatTemplateFromGguf"],
        usings=["LoadChatTemplateFromGguf"],
        notes=["恢复 test_chat.cc 和 test_gguf_tokenizer.cc 里标了 TODO(step20) 的用例。"],
        questions=["多轮对话时，历史 assistant 回复要不要带 <|im_end|>？"],
    ),
    21: dict(
        stage=4, title="真实模型端到端 Smoke", goal="跑通真实模型的最小闭环",
        verify=["./build/mini-llama run models/chat -n 64",
                "# 用中文和英文各问一个问题，回复要有语义"],
        notes=["如果输出乱码，按 tokenizer -> chat template -> 第 1 个 token 的 logits 的顺序逐个排查。"],
        questions=["记录一次完整对话里 prefill 和 decode 各自的耗时。"],
    ),
    22: dict(
        stage=5, title="Matmul Dispatch", goal="统一 naive、quant、threaded、SIMD 路径",
        read=["include/mini_llama/matmul_dispatch.h", "src/matmul_optimized.cc"],
        write=["include/mini_llama/matmul_dispatch.h", "src/matmul_optimized.cc", "src/main.cc (--threads)"],
        sources=["src/matmul_optimized.cc"],
        includes=["matmul_dispatch.h"],
        usings=["LinearDispatch", "MatmulDispatch", "MatmulMode"],
        verify=["./build/mini-llama generate --config models/tiny/model.json --model models/tiny/model.bin "
                "-n 4 --threads -1   # 应给出清晰的错误提示"],
        notes=["这一步先只接 naive 和 quant 两条路径，threaded 和 SIMD 在后两步补上。"],
        questions=["dispatch 层应该在哪里检查非法参数，为什么？"],
    ),
    23: dict(
        stage=5, title="Thread Pool + 按行并行 matmul", goal="实现 row-parallel matmul",
        read=["include/mini_llama/thread_pool.h", "src/thread_pool.cc", "src/matmul_optimized.cc"],
        write=["include/mini_llama/thread_pool.h", "src/thread_pool.cc", "matmul_optimized.cc 增加 threaded 路径"],
        sources=["src/thread_pool.cc"],
        includes=["thread_pool.h"],
        usings=["GetThreadCount", "ParallelFor", "SetThreadCount"],
        tests=["test_threaded_matmul.cc"],
        copy=["scripts/test_threaded_cli.py"],
        pytests=[("threaded-cli-smoke", "test_threaded_cli.py")],
        verify=[f"./build/mini-llama bench models/chat -n 16 --threads 1",
                f"./build/mini-llama bench models/chat -n 16 --threads 8"],
        notes=["test_threaded_matmul.cc 里 SIMD 相关用例先注释掉并标 // TODO(step24)。"],
        questions=["为什么按行切分不需要加锁？", "单线程和多线程的浮点结果为什么能做到完全一致？"],
    ),
    24: dict(
        stage=5, title="SIMD (AVX2)", goal="接入 AVX2 的最小优化路径，并保留通用 fallback",
        read=["src/matmul_optimized.cc", "src/quant.cc"],
        write=["matmul_optimized.cc / quant.cc 增加 AVX2 点积"],
        verify=["grep -o -m1 avx2 /proc/cpuinfo",
                f"./build/mini-llama bench models/chat -n 16 --threads 1"],
        notes=["用 #if defined(__AVX2__) 保护，并在 CMake 里按需加 -mavx2 -mfma。",
               "恢复 test_threaded_matmul.cc 里标了 TODO(step24) 的用例。"],
        questions=["SIMD 求和顺序改变后，为什么结果和 naive 版本会有微小差异？阈值怎么定？"],
    ),
    25: dict(
        stage=5, title="Python 参考对齐进阶", goal="对齐 F32、量化、多线程路径",
        read=["scripts/test_golden.py", "scripts/compare_logits.py"],
        verify=["./build/mini-llama generate --prompt hello -n 16 --threads 4 --dump-logits artifacts/t4",
                "python3 scripts/make_golden.py --config models/tiny/model.json --weights models/tiny/model.bin "
                "--prompt hello -n 16 --output-dir artifacts/t4",
                "python3 scripts/compare_logits.py --cpp-dir artifacts/t4 --py-dir artifacts/t4 --vocab-size 128"],
        notes=["可以扩展 test_golden.py，让它同时覆盖 F32、--quant q8_0 和 --threads 4 三种情况。"],
        questions=["量化路径的 logits 阈值应该放宽到多少？依据是什么？"],
    ),
    26: dict(
        stage=6, title="Backend Selection + CLI", goal="接入 --backend cpu|cuda、--device、--cuda-info",
        read=["include/mini_llama/backend.h", "src/backend.cc", "src/main.cc (backend 部分)"],
        write=["include/mini_llama/backend.h", "src/backend.cc", "src/main.cc"],
        sources=["src/backend.cc"],
        includes=["backend.h"],
        usings=["BackendConfig", "BackendExecutionNote", "BackendKind"],
        copy=["scripts/test_backend_cli.py"],
        pytests=[("backend-cli-smoke", "test_backend_cli.py")],
        verify=["./build/mini-llama generate --config models/tiny/model.json --model models/tiny/model.bin "
                "-n 4 --backend cuda   # CPU 构建下应给出明确报错"],
        questions=["sampler 为什么留在 CPU 上？"],
    ),
    27: dict(
        stage=6, title="CMake CUDA 开关 + CUDA Runtime stub", goal="掌握 MINI_LLAMA_CUDA 的构建边界",
        read=["CMakeLists.txt", "include/mini_llama/cuda_runtime.h", "src/cuda_runtime.cc"],
        write=["include/mini_llama/cuda_runtime.h", "src/cuda_runtime.cc (用 #ifdef MINI_LLAMA_USE_CUDA 分出 stub)"],
        sources=["src/cuda_runtime.cc"],
        includes=["cuda_runtime.h"],
        usings=["CudaDeviceBuffer", "CudaDeviceCount", "CudaDeviceInfo", "CudaFormatDeviceInfo",
                "CudaGetDeviceInfo", "CudaRuntimeBuilt", "CudaSetDevice"],
        cmake={"@options": CUDA_OPTIONS, "@core_link": CUDA_CORE_LINK,
               "@extra": cuda_test_exe("mini-llama-cuda-runtime-tests", "tests/test_cuda_runtime.cc",
                                       [("cuda_runtime", "")])},
        copy=["tests/test_cuda_runtime.cc"],
        verify=["./build/mini-llama --cuda-info   # CPU 构建下应提示未启用 CUDA"],
        notes=["test_cuda_runtime.cc 有自己的 main，在 CPU 构建下会断言 CudaRuntimeBuilt() == false。"],
        questions=["为什么 CPU 构建也要编译 cuda_runtime.cc？stub 的意义是什么？"],
    ),
    28: dict(
        stage=6, title="CUDA 接口先行", goal="写好所有 CUDA 头文件和 CPU 构建下的 stub",
        read=["include/mini_llama/cuda_tensor.h", "include/mini_llama/cuda_matmul.h",
              "include/mini_llama/cuda_quant.h", "include/mini_llama/cuda_ops.h",
              "include/mini_llama/cuda_kv_cache.h", "include/mini_llama/cuda_attention.h",
              "include/mini_llama/model.h", "include/mini_llama/context.h"],
        write=["上面 6 个 cuda_*.h 和对应的 src/cuda_*.cc", "model.h / model.cc / context.h 增加 CUDA 字段和统计"],
        sources=["src/cuda_tensor.cc", "src/cuda_matmul.cc", "src/cuda_quant.cc", "src/cuda_ops.cc",
                 "src/cuda_kv_cache.cc", "src/cuda_attention.cc"],
        includes=["cuda_tensor.h", "cuda_matmul.h", "cuda_quant.h", "cuda_ops.h", "cuda_kv_cache.h",
                  "cuda_attention.h"],
        usings=["CudaTensor", "CudaTensorFromHost", "CudaLinear", "CudaLinearDeviceInput", "CudaMatmul",
                "CudaMatmulBuilt", "CudaQ40Linear", "CudaQ40LinearDeviceInput", "CudaQ41Linear",
                "CudaQ41LinearDeviceInput", "CudaQ80Linear", "CudaQ80LinearDeviceInput", "CudaQuantBuilt",
                "CudaElementwiseAdd", "CudaElementwiseAddDeviceInput", "CudaElementwiseAddDeviceWeight",
                "CudaElementwiseMul", "CudaElementwiseMulDeviceInput", "CudaEmbeddingLookupDeviceWeight",
                "CudaOpsBuilt", "CudaRmsNorm", "CudaRmsNormDeviceInput", "CudaRmsNormDeviceWeight",
                "CudaSilu", "CudaSiluDeviceInput", "CudaKvCache", "CudaKvCacheBuilt", "CudaAttentionBuilt",
                "CudaAttentionDecode", "CudaAttentionDecodeDeviceInput",
                "ModelCudaActivationCalls", "ModelCudaAttentionCalls", "ModelCudaAttentionCpuFallbacks",
                "ModelCudaDeviceToHostBytes", "ModelCudaDeviceToHostCopies", "ModelCudaHostToDeviceBytes",
                "ModelCudaHostToDeviceCopies", "ModelCudaKvCacheReadBytes", "ModelCudaKvCacheWriteBytes",
                "ModelCudaLinearCalls", "ModelCudaMemoryBytes", "ModelCudaUploadedWeightCount",
                "ResetModelCudaRuntimeStats", "UploadModelWeightsToCuda"],
        copy=["tests/test_cuda_matmul.cc", "tests/test_cuda_quant.cc", "tests/test_cuda_forward_buffer.cc",
              "tests/test_cuda_attention.cc", "tests/test_cuda_forward_linear.cc", "tests/test_cuda_ops.cc",
              "tests/test_cuda_kv_cache.cc", "tests/test_cuda_forward.cc"],
        cmake={"@extra": (
            cuda_test_exe("mini-llama-cuda-matmul-tests", "tests/test_cuda_matmul.cc",
                          [("cuda_matmul", "matmul"), ("cuda_linear", "linear")])
            + cuda_test_exe("mini-llama-cuda-quant-tests", "tests/test_cuda_quant.cc",
                            [("cuda_quant_q8_0", "q8_0"), ("cuda_quant_q4", "q4"),
                             ("cuda_forward_quant", "forward"), ("cuda_forward_q4", "forward_q4")])
            + cuda_test_exe("mini-llama-cuda-forward-buffer-tests", "tests/test_cuda_forward_buffer.cc",
                            [("cuda_forward_buffer", "f32"), ("cuda_forward_device_resident", "q8_0")])
            + cuda_test_exe("mini-llama-cuda-attention-tests", "tests/test_cuda_attention.cc",
                            [("cuda_attention", "direct"), ("cuda_kv_attention", "kv"),
                             ("cuda_forward_attention", "forward")])
            + cuda_test_exe("mini-llama-cuda-forward-linear-tests", "tests/test_cuda_forward_linear.cc",
                            [("cuda_forward_linear", "")])
            + cuda_test_exe("mini-llama-cuda-ops-tests", "tests/test_cuda_ops.cc", [("cuda_ops", "")])
            + cuda_test_exe("mini-llama-cuda-kv-cache-tests", "tests/test_cuda_kv_cache.cc",
                            [("cuda_kv_cache", "")])
            + cuda_test_exe("mini-llama-cuda-forward-tests", "tests/test_cuda_forward.cc",
                            [("cuda_forward", "")]))},
        notes=["这些 CUDA 测试在 CPU 构建下只验证“未启用 CUDA 时行为正确”，真正的数值验证要等有 GPU。",
               "完成后 tests/test_names.h 应和原项目的版本等价。"],
        questions=["CudaDeviceBuffer 为什么要禁止拷贝、只允许移动？"],
    ),
    29: dict(
        stage=6, title="CUDA kernel 阅读与重写", goal="按依赖顺序重写 .cu kernel 和 device-resident decode",
        read=["src/cuda_runtime.cc", "src/cuda_tensor.cc", "src/cuda_matmul.cc", "src/cuda_quant.cu",
              "src/cuda_ops.cu", "src/cuda_kv_cache.cc", "src/cuda_attention.cu", "src/forward.cc (CUDA 分支)"],
        write=["src/cuda_quant.cu", "src/cuda_ops.cu", "src/cuda_attention.cu", "以及各 cuda_*.cc 的真实实现"],
        cmake={"@cuda_sources": [
            "if(MINI_LLAMA_CUDA)",
            "    list(APPEND MINI_LLAMA_CORE_SOURCES src/cuda_attention.cu src/cuda_ops.cu src/cuda_quant.cu)",
            "endif()"]},
        create=["src/cuda_attention.cu", "src/cuda_ops.cu", "src/cuda_quant.cu"],
        notes=["顺序: Runtime -> DeviceBuffer/CudaTensor -> cuBLAS F32 linear -> 权重上传 -> 量化 kernel -> "
               "激活 kernel -> GPU KV cache -> attention -> device-resident decode -> fallback 统计。",
               "没有 GPU 时，可以先为每个 kernel 写一个 CPU 版的 grid/block 模拟循环，帮助理解下标计算。",
               ".cu 文件只在 -DMINI_LLAMA_CUDA=ON 时编译，CPU 构建不受影响。"],
        questions=["RMSNorm kernel 里 block 内求和用的是什么归约方式？",
                   "decode attention 每个 block 负责什么？"],
    ),
    30: dict(
        stage=6, title="云 GPU 验证与 Benchmark 报告", goal="在 NVIDIA GPU 上跑通全部 CUDA 测试并产出报告",
        read=["scripts/run_cuda_benchmarks.py", "scripts/test_cuda_chat_smoke.py"],
        copy=["scripts/run_cuda_benchmarks.py", "scripts/test_cuda_chat_smoke.py", "scripts/test_cuda_asan.sh"],
        pytests=[("cuda-chat-smoke", "test_cuda_chat_smoke.py")],
        verify=["nvidia-smi && nvcc --version",
                "cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-cuda -j",
                "ctest --test-dir build-cuda --output-on-failure",
                "./build-cuda/mini-llama bench models/chat --backend cuda -p hello -n 32 --seed 1",
                "python3 scripts/run_cuda_benchmarks.py"],
        notes=["没有 GPU 时，这一步只需确认 CPU 构建下 cuda-chat-smoke 能正常跳过。"],
        questions=["报告里 H2D / D2H 的字节数说明了 decode 主链路是否真正 device-resident？"],
    ),
}


# ---------------------------------------------------------------------------

def is_valid_gguf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"started": [], "done": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def insert_before_marker(path, marker, lines, atomic=False):
    """atomic=True 时把 lines 当作一个整体片段，已存在则整体跳过。"""
    text = path.read_text(encoding="utf-8")
    existing = {line.strip() for line in text.splitlines()}
    if atomic:
        block = "\n".join(line.strip() for line in lines)
        stripped = "\n".join(line.strip() for line in text.splitlines())
        if not lines or block in stripped:
            return []
    out = []
    added = []
    for raw in text.splitlines(keepends=True):
        if raw.strip() == f"# {marker}" or raw.strip() == f"// {marker}":
            indent = raw[: len(raw) - len(raw.lstrip())]
            for line in lines:
                if line.strip() and (atomic or line.strip() not in existing):
                    out.append(indent + line + "\n")
                    added.append(line.strip())
        out.append(raw)
    if added:
        path.write_text("".join(out), encoding="utf-8")
    return added


def copy_from_ref(rel):
    src = REF / rel
    dst = ROOT / rel
    if dst.exists():
        return f"  跳过（已存在）: {rel}"
    if not src.exists():
        return f"  警告: 原项目中找不到 {rel}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return f"  拷贝: {rel}"


def print_step(n, step):
    state = load_state()
    flag = "完成" if n in state["done"] else ("进行中" if n in state["started"] else "未开始")
    print(f"\n=== 第 {n} 步 [阶段 {step['stage']}] {step['title']}  ({flag}) ===")
    print(f"目标: {step['goal']}")
    if step.get("read"):
        print("\n建议阅读（原项目，卡住超过 30 分钟再看 src 实现）:")
        for f in step["read"]:
            print(f"  {REF / f.split(' ')[0]}" + (f"  {' '.join(f.split(' ')[1:])}" if ' ' in f else ""))
    if step.get("write"):
        print("\n你需要自己写:")
        for f in step["write"]:
            print(f"  {f}")
    if step.get("tests") or step.get("pytests"):
        print("\n验收测试:")
        for t in step.get("tests", []):
            print(f"  tests/{t}")
        for name, script in step.get("pytests", []):
            print(f"  ctest -R {name}  (scripts/{script})")
    print("\n通用验收: python3 scripts/step.py check")
    if step.get("verify"):
        print("\n手动验收:")
        for v in step["verify"]:
            print(f"  {v}")
    if step.get("notes"):
        print("\n注意:")
        for note in step["notes"]:
            print(f"  - {note}")
    print()


def cmd_list():
    state = load_state()
    current = None
    for n, step in STEPS.items():
        if current != step["stage"]:
            current = step["stage"]
            print(f"\n阶段 {current}")
        mark = "[x]" if n in state["done"] else ("[~]" if n in state["started"] else "[ ]")
        print(f"  {mark} {n:2d}. {step['title']}")
    pending = [n for n in STEPS if n not in state["done"]]
    if pending:
        print(f"\n下一步: python3 scripts/step.py start {pending[0]}")
    print()


def cmd_start(n):
    step = STEPS[n]
    if not REF.exists():
        sys.exit(f"找不到原项目: {REF}（可设置环境变量 MINI_LLAMA_REF）")
    print(f"进入第 {n} 步: {step['title']}")

    for rel in step.get("copy", []):
        print(copy_from_ref(rel))
    for t in step.get("tests", []):
        print(copy_from_ref(f"tests/{t}"))

    for rel in step.get("sources", []) + step.get("create", []):
        p = ROOT / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("// SPDX-License-Identifier: MIT\n", encoding="utf-8")
            print(f"  创建空文件: {rel}")

    added = insert_before_marker(CMAKE_FILE, "@sources", step.get("sources", []))
    added += insert_before_marker(CMAKE_FILE, "@tests", [f"tests/{t}" for t in step.get("tests", [])])
    added += insert_before_marker(CMAKE_FILE, "@pytests",
                                  [pytest_block(name, s) for name, s in step.get("pytests", [])])
    for marker, lines in step.get("cmake", {}).items():
        added += insert_before_marker(CMAKE_FILE, marker, lines, atomic=True)
    if added:
        print(f"  CMakeLists.txt: 新增 {len(added)} 行")

    inc = insert_before_marker(NAMES_FILE, "@includes",
                               [f'#include "mini_llama/{h}"' for h in step.get("includes", [])])
    use = insert_before_marker(NAMES_FILE, "@usings",
                               [f"using mini_llama::{s};" for s in step.get("usings", [])])
    if inc or use:
        print(f"  tests/test_names.h: 新增 {len(inc)} 个 include, {len(use)} 个 using")

    if step.get("link_chat_model"):
        link = ROOT / CHAT_GGUF
        target = REF / CHAT_GGUF
        if is_valid_gguf(link):
            print(f"  真实模型已就绪: {CHAT_GGUF}")
        elif is_valid_gguf(target) and not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)
            print(f"  软链接: {CHAT_GGUF} -> {target}")
        else:
            print("  真实模型不存在（原项目里只有 Git LFS 指针），请先下载:")
            print("    python3 scripts/download_demo_model.py")

    notes_dir = ROOT / "notes"
    notes_dir.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "_-" else "-" for c in step["title"])
    slug = "-".join(part for part in slug.split("-") if part)
    note = notes_dir / f"{n:02d}-{slug}.md"
    if not note.exists():
        body = [f"# 第 {n} 步: {step['title']}", "", f"目标: {step['goal']}", "", "## 思考题", ""]
        for q in step.get("questions", []):
            body += [f"- {q}", "", "  答: ", ""]
        body += ["## 数据流 / shape / 边界", "", "## 踩过的坑", ""]
        note.write_text("\n".join(body), encoding="utf-8")
        print(f"  笔记模板: {note.relative_to(ROOT)}")

    state = load_state()
    if n not in state["started"]:
        state["started"].append(n)
        save_state(state)
    print_step(n, step)


def cmd_check():
    build = ROOT / "build"
    cmds = [
        ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Debug"],
        ["cmake", "--build", str(build), "-j"],
        ["ctest", "--test-dir", str(build), "--output-on-failure"],
    ]
    for c in cmds:
        print("$ " + " ".join(c))
        if subprocess.run(c, cwd=ROOT).returncode != 0:
            print("\ncheck 失败，请根据上面的输出修复。")
            return False
    print("\ncheck 通过。")
    return True


def cmd_done(n):
    if not cmd_check():
        sys.exit(1)
    state = load_state()
    if n not in state["done"]:
        state["done"].append(n)
        state["done"].sort()
        save_state(state)
    print(f"第 {n} 步已标记为完成。")
    pending = [k for k in STEPS if k not in state["done"]]
    if pending:
        print(f"下一步: python3 scripts/step.py start {pending[0]}")


def parse_step(arg):
    try:
        n = int(arg)
    except ValueError:
        sys.exit(f"步骤编号必须是整数: {arg}")
    if n not in STEPS:
        sys.exit(f"步骤编号范围是 1-{max(STEPS)}")
    return n


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd = argv[1]
    if cmd == "list":
        cmd_list()
    elif cmd == "check":
        sys.exit(0 if cmd_check() else 1)
    elif cmd in ("show", "start", "done") and len(argv) == 3:
        n = parse_step(argv[2])
        {"show": lambda: print_step(n, STEPS[n]), "start": lambda: cmd_start(n),
         "done": lambda: cmd_done(n)}[cmd]()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
