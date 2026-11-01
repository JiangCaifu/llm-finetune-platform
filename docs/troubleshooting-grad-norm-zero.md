# 踩坑记录：grad_norm=0，模型完全不学习

## 问题现象

QLoRA 微调 Qwen2-7B-Instruct，训练能跑完 3 个 epoch（22 分钟），但**模型完全没有学到任何东西**：

```
{'loss': 2.3738, 'grad_norm': 0.0, 'mean_token_accuracy': 0.5098, 'epoch': 0.29}
{'loss': 2.378,  'grad_norm': 0.0, 'mean_token_accuracy': 0.5095, 'epoch': 0.57}
{'loss': 2.4803, 'grad_norm': 0.0, 'mean_token_accuracy': 0.4929, 'epoch': 0.86}
```

三个铁证：
1. **`grad_norm` 全程为 0.0**（正常应在 0.1~10 之间）
2. **`loss` 不下降**（2.37 → 2.41，甚至略微上升）
3. **三轮 `eval_loss` 完全相同**（2.4235 → 2.4235 → 2.4235，模型参数根本没变）

## 根因

**gradient checkpointing 的 `use_reentrant=True`（旧默认模式）与 LoRA 不兼容。**

### 旧 reentrant 模式的致命逻辑

reentrant checkpoint 通过"checkpoint 区域的输入是否 `requires_grad`"来决定是否构建反向图：

```
输入 require_grad?
  ├─ 是 → 建反向图，梯度能回传
  └─ 否 → 完全不建反向图
```

LoRA 冻结了 base model，所以：

```
token ids
  └─ embedding 层（冻结，requires_grad=False）
      └─ embedding 输出 ← requires_grad=False
          └─ checkpoint 区域 ← 看到输入不需要 grad → 不建反向图
              └─ LoRA 的 A/B 矩阵（requires_grad=True，但反向图没建到这）
                  └─ grad_norm = 0.0
```

**LoRA 参数明明 `requires_grad=True`，但反向图根本没构建到它们** —— 梯度无路可走。

### 正常的 non-reentrant 模式

```
token ids
  └─ embedding 层（冻结）
      └─ embedding 输出 ← enable_input_require_grads 强制 require_grad=True
          └─ checkpoint 区域 ← non-reentrant 模式，为 LoRA 参数建反向图
              └─ LoRA 的 A/B 矩阵 ← 梯度正常回传
                  └─ grad_norm = 9.54（正常）
```

## 诊断验证

写了 `scripts/diagnose_gradient.py`，对同一样本跑 forward+backward，对比不同配置的实际 `grad_norm`：

| 配置 | use_reentrant | enable_input_require_grads | grad_norm | 结论 |
|---|---|---|---|---|
| A（标准修复） | False | 是 | **9.537** | ✅ 正常 |
| B | True | 是 | 崩溃（显存问题） | 脚本缺陷 |
| C（复现 bug） | True | 否 | 未测到 | — |
| D（4-bit） | True（prepare_model） | 自动 | 未测到 | — |

配置 A 的成功已经确认修复方向：`use_reentrant=False` + `enable_input_require_grads` 产生正常梯度。

## 修复

`scripts/finetune_qlora.py` 模型加载部分（3 行关键改动）：

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False

# 关键 1：use_reentrant=False（非 reentrant 模式）
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# 关键 2：让 embedding 输出 require grad（双保险）
model.enable_input_require_grads()
```

SFTConfig 里也保持一致：

```python
sft_config = SFTConfig(
    ...
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ...
)
```

## 修复后预期

训练前 30 步就能看到：
- `grad_norm` 非零（0.1~10 之间）
- `loss` 从 ~2.3 开始下降
- `mean_token_accuracy` 从 0.50 上升到 0.6+

## 为什么旧模式这么设计

旧 reentrant 模式是 PyTorch 早期的 checkpoint 实现，假设是"如果输入不需要 grad，里面参数也不需要更新，跳过建图省时间"。这个假设在**全参数训练时成立**（所有参数都要 grad）。但 **LoRA 打破了这个假设** —— base 冻结但 LoRA 参数要更新。

PyTorch 后来出了 non-reentrant 模式修复这个问题，但 transformers 的 `gradient_checkpointing_enable()` 默认还是 `use_reentrant=True`（向后兼容）。这就是为什么很多人踩这个坑。

## 参考资料

- [PEFT + gradient checkpointing 已知问题](https://www.sabrresearch.com/cookbooks/lora-gradient-checkpointing-detached-tensor-fix)
- [use_reentrant 参数说明](https://www.netray.co/troubleshoot/gradient-checkpointing-errors)
- [训练指标健康范围（axolotl 文档）](https://docs.axolotl.ai/docs/training_stability.html)

## 踩过的其他坑（按时间顺序）

### 1. ChatML 模板未设置
- **现象**：`tokenizer.chat_template is not set`
- **修复**：手动构造 ChatML 模板字符串，不依赖 `tokenizer.chat_template`
- **commit**：`2ed4050`

### 2. LoRA target_modules 名字不对（Qwen-7B-Chat）
- **现象**：`{'gate_proj', 'up_proj', 'down_proj', ...} not found`
- **修复**：Qwen-7B-Chat 用旧命名 `c_attn/c_proj/w1/w2`，Qwen2-7B-Instruct 用标准命名 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`
- **最终方案**：迁移到 Qwen2-7B-Instruct，用标准命名
- **commit**：`1bd4448` → `d2b4167`

### 3. trl 0.22 SFTConfig API 变化
- **现象**：`unexpected keyword argument 'max_seq_length'`
- **修复**：`max_seq_length` → `max_length`，所有参数直接在 SFTConfig 设置
- **commit**：`4faae51`

### 4. pad_token 为 None（Qwen-7B-Chat）
- **现象**：`'NoneType' object is not iterable`
- **修复**：Qwen-7B-Chat 的 tokenizer_config.json 缺特殊 token 配置，需手动补齐
- **最终方案**：迁移到 Qwen2-7B-Instruct（tokenizer 配置完整，无需手动补齐）
- **commit**：`4294c17` → `9219ae5` → `d2b4167`

### 5. Flash Attention dtype 不匹配（Qwen-7B-Chat）
- **现象**：`assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))`
- **修复**：Qwen-7B-Chat 的 flash attention 实现在 bf16 下有问题
- **最终方案**：迁移到 Qwen2-7B-Instruct（原生支持 bf16）
- **commit**：`c027839` → `d2b4167`

### 6. grad_norm=0（本文档）
- **现象**：训练能跑完但模型完全不学习
- **根因**：gradient checkpointing 的 reentrant 模式与 LoRA 不兼容
- **修复**：`use_reentrant=False` + `enable_input_require_grads`
- **commit**：待提交

## 经验总结

1. **grad_norm=0 是铁定异常** —— 正常训练的 grad_norm 应在 0.1~10 之间，0.0 说明梯度根本没回传
2. **eval_loss 多轮完全相同** = 模型参数没变 = 训练完全无效
3. **mean_token_accuracy 卡在 0.50** = 随机猜测水平 = 模型没学到
4. **遇到不学习的训练，先写诊断脚本测 forward+backward 的实际 grad_norm**，不要盲目调参重训
5. **LoRA + gradient checkpointing 必须设 `use_reentrant=False`** —— 这是 PEFT 的经典坑
