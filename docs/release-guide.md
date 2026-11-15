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

1. 下载 `adapter_model.safetensors` 和 `adapter_config.json`
2. 放入 `outputs/qwen-tourism-lora/` 目录
3. 启动服务：

```bash
docker compose up -d
```

## 文件说明

| 文件 | 说明 |
|------|------|
| adapter_model.safetensors | LoRA 权重 (80MB) |
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

```bash
cd /mnt/workspace/llm-finetune-platform
tar czf lora-weights.tar.gz -C outputs/qwen-tourism-lora \
    adapter_config.json \
    adapter_model.safetensors \
    tokenizer_config.json \
    tokenizer.json \
    vocab.json \
    merges.txt

# 查看大小
ls -lh lora-weights.tar.gz
```

### 3. 下载到本地

**方式 A：DSW JupyterLab 文件管理器**
- 在 JupyterLab 左侧文件浏览器找到 `lora-weights.tar.gz`
- 右键 → Download

**方式 B：split + base64（如果文件太大）**
```bash
# DSW 上：分割成小文件
split -b 50M lora-weights.tar.gz lora-part-
# 下载后本地合并
cat lora-part-* > lora-weights.tar.gz
```

### 4. 本地解压 + 上传 GitHub Release

```bash
# 本地解压
tar xzf lora-weights.tar.gz -C outputs/qwen-tourism-lora/

# 然后在 GitHub Release 页面上传以下文件：
# - adapter_model.safetensors
# - adapter_config.json
# - tokenizer_config.json
# - tokenizer.json
# - vocab.json
# - merges.txt
```

### 5. 验证

下载 Release 文件，放入 `outputs/qwen-tourism-lora/`，启动 Docker 验证服务正常。
