# GitHub Release 草稿说明

## 创建 Release 的步骤

1. 在 GitHub 仓库页面点击 **Releases** → **Create a new release**
2. Tag: `v1.0`（或 `v1.0-lora-weights`）
3. 标题: `LoRA 权重 v1.0 - Qwen2-7B-Instruct 文旅微调`
4. 描述见下方
5. 上传文件（从 DSW 下载后）
6. 发布

---

## Release 描述草稿

```
# LoRA 权重 v1.0

## 基本信息

- 基座模型：Qwen2-7B-Instruct
- 微调方法：LoRA (PEFT)
- 训练数据：700 条文旅指令数据（560 train / 70 val / 70 test）
- 训练参数：3 epoch, rank=8, alpha=32, bf16, AdamW
- 可训练参数：20,185,088 (0.26%)

## 训练效果

| 指标 | 值 |
|------|-----|
| Train Loss | 1.20 |
| Eval Loss | 1.22 (best at epoch 2) |
| Mean Token Accuracy | 0.75 |
| 训练时长 | 16 分钟 (A10 GPU) |

## 使用方式

### 准备基座模型

LoRA 权重只是「差量」，必须配合基座模型使用。下载 Qwen2-7B-Instruct 后，目录结构建议如下：

```
<项目根目录>/
├── models/Qwen2-7B-Instruct/     # 基座模型完整目录
└── outputs/qwen-tourism-lora/     # 本 Release 的权重解压到这里
```

### 放置权重

1. 下载 `adapter_model.safetensors` 和 `adapter_config.json`
2. 解压放入 `outputs/qwen-tourism-lora/` 目录

### 启动服务

```bash
# 若基座模型不在默认位置，先指定路径
export BASE_MODEL_PATH=/path/to/models/Qwen2-7B-Instruct

docker compose up -d
curl http://localhost:8000/api/v1/health
```

健康检查返回 `"model_loaded": true` 后即可访问前端页面。

## 文件说明

| 文件 | 说明 |
|------|------|
| adapter_model.safetensors | LoRA 权重 (78MB) |
| adapter_config.json | LoRA 配置 |
| tokenizer_config.json | Tokenizer 配置 |
| tokenizer.json | Tokenizer 词表 |
| vocab.json | BPE 词表 |
| merges.txt | BPE 合并规则 |

## 注意事项

- 需要单独下载 Qwen2-7B-Instruct 基座模型
- GPU 显存要求：16GB+ (bf16 推理)
- 此权重仅用于文旅问答场景，不保证其他领域效果
```

---

## 从 DSW 下载权重的步骤

### 1. 开 DSW 实例

### 2. 打包权重

⚠️ **必须排除 `checkpoint-*` 和 `runs/`**。开启 `load_best_model_at_end` 后，每个 checkpoint 除了 adapter 权重还包含
AdamW 优化器状态（20,185,088 参数 × 2 states × fp32 ≈ 161MB），3 个 checkpoint 会让压缩包膨胀到 **726MB**，
而推理根本用不到这些中间产物。

```bash
cd /mnt/workspace/llm-finetune-platform/outputs/qwen-tourism-lora

# 确认各子目录实际占用（ls 显示的目录大小只是 inode，不反映内容）
du -sh checkpoint-* runs

# 打包顶层文件，排除 checkpoint 与 TensorBoard 事件目录
tar czf /mnt/workspace/qwen-tourism-lora.tar.gz \
    --exclude='checkpoint-*' \
    --exclude='runs' \
    .

# 校验包内清单 + 记录校验和
tar tzf /mnt/workspace/qwen-tourism-lora.tar.gz
ls -lh /mnt/workspace/qwen-tourism-lora.tar.gz
md5sum /mnt/workspace/qwen-tourism-lora.tar.gz
```

预期体积 **~85MB**（`adapter_model.safetensors` 78MB + tokenizer 约 15MB）。

### 3. 下载到本地

**方式 A：DSW JupyterLab 文件管理器**
- 在 JupyterLab 左侧文件浏览器找到 `qwen-tourism-lora.tar.gz`
- 右键 → Download

**方式 B：split 分片（下载中断时使用）**
```bash
# DSW 上：分割成小文件
split -b 50M /mnt/workspace/qwen-tourism-lora-min.tar.gz /mnt/workspace/lora-part-
# 下载后本地合并
cat lora-part-* > qwen-tourism-lora-min.tar.gz
```

### 4. 本地校验 + 解压

```powershell
# 比对 MD5（必须与 DSW 上 md5sum 的输出一致）
Get-FileHash qwen-tourism-lora-min.tar.gz -Algorithm MD5

# 一致后才解压
# 注意：tar 是在 qwen-tourism-lora/ 目录内部打包的（tar czf ... .），包内是平铺结构，
#       因此解压目标必须是 outputs/qwen-tourism-lora/，否则会平铺到 outputs/ 下
tar xzf qwen-tourism-lora-min.tar.gz -C C:\Users\zhaoxi\Projects\llm-finetune-platform\outputs\qwen-tourism-lora\
```

解压后确认 `outputs\qwen-tourism-lora\adapter_model.safetensors` 存在且约 78MB。

### 5. 上传 GitHub Release

在 GitHub Release 页面上传 `outputs/qwen-tourism-lora/` 下的文件：

| 文件 | 必需 | 说明 |
|------|------|------|
| adapter_model.safetensors | ✅ | LoRA 权重本体（78MB） |
| adapter_config.json | ✅ | LoRA 配置（rank/alpha/target_modules） |
| tokenizer_config.json | 可选 | Tokenizer 配置 |
| tokenizer.json | 可选 | Tokenizer 词表（11MB） |
| vocab.json | 可选 | BPE 词表 |
| merges.txt | 可选 | BPE 合并规则 |
| special_tokens_map.json | 可选 | 特殊 token 映射 |
| added_tokens.json | 可选 | 新增 token |
| chat_template.jinja | 可选 | ChatML 模板 |

> tokenizer 相关文件实际可从基座模型获得，此处打包是为了让权重包可独立分发。

### 6. 验证

1. 从 Release 下载文件放入 `outputs/qwen-tourism-lora/`
2. 启动服务（本地 GPU 环境）
   ```bash
   docker compose up -d
   curl http://localhost:8000/api/v1/health
   ```
3. 确认 `model_loaded: true` 且 `/api/v1/chat` 返回文旅领域答案

### 7. 清理 DSW 存储（可选）

本地校验通过后可删除 DSW 上的压缩包以释放存储配额：

```bash
rm -f /mnt/workspace/qwen-tourism-lora.tar.gz
```

⚠️ 只点 DSW 的「停止」，**不要点「释放」**——释放会清空 `/mnt/workspace` 工作区。
