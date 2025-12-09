# MindNLP 昇腾推理优化说明

## 优化概述

本次提交针对 **Qwen2-VL**、**Janus-Pro** 模型进行了深度优化。主要策略聚焦于**核心算子融合 (Kernel Fusion)**、**计算图精简**、**精度管理策略**以及**昇腾亲和算子替换**，旨在显著提升 NPU 上的推理吞吐量并降低显存开销。

## 核心优化详解

### 1. Qwen2-VL

**核心算子融合 (Fused Kernels)**

- **RoPE 旋转位置编码融合**:
    - **优化前**: 使用 Python 循环进行复杂的切片 (`split`)、拼接 (`cat`) 和手动乘法计算，导致大量细碎算子下发，Host-Device 交互频繁。
    - **优化后**:
      ```python
      # 替换手动实现，直接调用昇腾融合算子
      q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin, mode=0)
      ```
    - **收益**: 将计算下沉至底层 C++ 算子，大幅减少 Kernel Launch 开销，显著提升 Encode 和 Decode 阶段效率。

- **RMSNorm 融合**:
    - **优化前**: 手动执行 `Mean` -> `Pow` -> `Rsqrt` -> `Mul` 序列。
    - **优化后**:
      ```python
      return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
      ```
    - **收益**: 利用 CANN 库优化实现，减少内存读写带宽占用。

**张量操作与显存优化**

- **Attention Mask 生成重构**:
    - **策略**: 重写 `_prepare_4d_causal_attention_mask`，使用 `ops.broadcast_to`、`ops.narrow` 和 `ops.masked_fill` 替代复杂的广播和切片赋值。
    - **收益**: 优化了 Mask 的内存布局（Contiguous），避免非连续内存访问带来的性能损耗。
- **算子替换**:
    - 将 Python 切片（如 `x[..., :half]`）替换为 `ops.split`。
    - 将 `swapaxes` 统一替换为 `ops.transpose`。
    - **收益**: 显式算子更能触发编译器的特定优化路径。

### 2.  Janus-Pro

**精度与计算流优化**

- **Softmax 去 FP32**:
    - **优化前**: `nn.functional.softmax(..., dtype=mindspore.float32)` 强制转 FP32 计算。
    - **优化后**:
      ```python
      attn_weights = nn.functional.softmax(attn_weights, dim=-1)
      ```
    - **收益**: 保持 BFloat16/FP16 数据流，避免不必要的 Cast 操作，减少显存带宽压力。

- **RoPE 参数预处理**:
    - **策略**: 将 `cos`/`sin` 的维度调整（`unsqueeze`）移至 Decoder 层循环之外，仅计算一次。
    - **收益**: 减少每层重复的 View/Reshape 操作，精简计算图。

- **Janus-Pro Tokenizer**:
    - 将 `vocab.get(tag)` 替换为 `convert_tokens_to_ids(tag)`，提高对不同 Tokenizer 实现的兼容性与鲁棒性。

- **Core SDPA (Scaled Dot Product Attention)**:
    - 明确转置维度：将 `key.swapaxes(-2, -1)` 优化为 `ops.transpose(key, (0, 1, 3, 2))`，辅助编译器进行内存排布优化。

## 技术亮点

1.  **激进的算子融合**: 全面启用 `rotary_position_embedding` 和 `rms_norm` 等昇腾专用融合算子，这是大模型推理加速的关键手段。
2.  **全链路半精度**: 移除 Attention 内部冗余的 FP32 类型转换，使得推理过程尽可能保持在 BF16/FP16，最大化利用 NPU 算力。
3.  **显式算子优先**: 大量使用 `ops.narrow`、`ops.split` 替代 Python 原生切片语法，减少隐式拷贝，使计算图对 NPU 编译器更友好。

## 优化收益预期

| 模块 | 优化点 | 预期收益 |
| :--- | :--- | :--- |
| **Qwen2-VL** | RoPE Fusion + RMSNorm | **推理延时显著降低**，长序列支持能力提升 |
| **Janus-Pro** | Softmax FP16/BF16 | **显存带宽占用减少**，Attention 计算吞吐提升 |
| **全局** | Transpose/Narrow 替换 | 减少算子编译时间，提升图执行效率 |

## 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 116.6667 |
| Prefill时延得分 | 287.3761    |
| Decode时延得分 | 190.0289     |
| **总分** | **198.0239** |
