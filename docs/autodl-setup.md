# AutoDL GPU 实例配置指南

## 1. 实例选择

| 配置 | 推荐 | 备选 | 说明 |
|------|------|------|------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) / A10 (24GB) | 7B+QLoRA 训练 24G 足够 |
| 镜像 | PyTorch 2.1 + CUDA 11.8 | PyTorch 2.0+ | 必须带 CUDA 11.8+ |
| CPU | 8核+ | 4核 | 数据加载需要 |
| 内存 | 32GB+ | 16GB | 加载模型需要 |
| 数据盘 | 50GB+ | 30GB | 模型权重 14GB + 输出 5GB |

## 2. 创建实例

1. 访问 https://www.autodl.com 注册并充值 100 元
2. "算力市场" → 选择 RTX 3090 节点（约 1.5元/小时）
3. "配置" → 选 **PyTorch 2.1.0 + Python 3.10 + CUDA 11.8**
4. 数据盘配置 50GB
5. 创建实例，记录 SSH 信息

## 3. SSH 连接实例

```bash
# AutoDL 给的连接信息（示例）
ssh -p 12345 root@connect.westc.gpuhub.com
# 输入密码登录
```

## 4. 环境准备

```bash
# 切换工作目录
cd /root

# 安装 git-lfs（用于下载大模型）
apt update && apt install -y git-lfs
git lfs install

# 配置 HuggingFace 镜像（加速下载）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 预下载 Qwen-7B-Chat 到本地（避免训练时下载失败）
huggingface-cli download Qwen/Qwen-7B-Chat --local-dir /root/models/qwen-7b-chat
```

## 5. 上传项目文件

**方式 A：用 AutoDL 提供的文件上传**

**方式 B：用 SCP 上传（本地 PowerShell）**

```powershell
# 在本地 PowerShell 中执行
# 上传项目代码（不含 .venv 和 outputs）
scp -P 12345 -r `
  c:\Users\zhaoxi\Projects\llm-finetune-platform\app `
  c:\Users\zhaoxi\Projects\llm-finetune-platform\scripts `
  c:\Users\zhaoxi\Projects\llm-finetune-platform\data `
  c:\Users\zhaoxi\Projects\llm-finetune-platform\requirements-gpu.txt `
  root@connect.westc.gpuhub.com:/root/llm-finetune-platform/
```

**方式 C：用 Git 仓库（推荐）**

先把项目推到 GitHub 私有仓库，AutoDL 上 clone 下来。

## 6. 安装依赖

```bash
cd /root/llm-finetune-platform

# 注意：torch 已随镜像安装，不要用 pip 默认源覆盖
pip install -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "import transformers, peft, trl; print(f'transformers: {transformers.__version__}, peft: {peft.__version__}, trl: {trl.__version__}')"
```

## 7. 开始训练

```bash
cd /root/llm-finetune-platform

# 启动训练
python scripts/finetune_qlora.py

# 后台运行 + 日志（推荐）
nohup python -u scripts/finetune_qlora.py > train.log 2>&1 &
tail -f train.log

# 监控显存（另开终端）
watch -n 5 nvidia-smi

# 监控 loss（另开终端）
tensorboard --logdir outputs/qwen-tourism-lora --port 6006
```

## 8. 评测

```bash
# 训练完成后
python scripts/evaluate_finetuned.py --samples 50

# 合并 LoRA（可选，部署前必做）
python scripts/merge_lora.py
```

## 9. 下载产物到本地

**方式 A：SCP 下载**

```powershell
# 本地 PowerShell
scp -P 12345 -r `
  root@connect.westc.gpuhub.com:/root/llm-finetune-platform/outputs/qwen-tourism-lora `
  c:\Users\zhaoxi\Projects\llm-finetune-platform\outputs\
```

**方式 B：AutoDL 文件管理**

实例详情页 → "JupyterLab" → 在文件管理器中下载。

## 10. 关机

训练完成后**立即关机**，按小时计费。

- 实例列表 → "关机"
- "关机"不收费，"释放"才删盘

## 11. 常见问题

### Q1: 训练显存不足（OOM）

```bash
# 减小 batch size
python scripts/finetune_qlora.py --batch_size 2 --grad_accum 8

# 减小 max_seq_length
python scripts/finetune_qlora.py --max_seq_length 384
```

### Q2: 模型下载失败

```bash
# 使用 hf-mirror
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen-7B-Chat --local-dir ./models/qwen-7b-chat
# 训练时指定本地路径
python scripts/finetune_qlora.py --model ./models/qwen-7b-chat
```

### Q3: 训练 loss 不下降

- 检查数据格式：必须是 Qwen ChatML 格式
- 调低学习率：`--lr 1e-4`
- 增加 warmup：修改脚本中 `warmup_ratio=0.1`

### Q4: 训练中断

```bash
# 从 checkpoint 恢复
python scripts/finetune_qlora.py --resume_from outputs/qwen-tourism-lora/checkpoint-100
```

## 12. 费用预估

| 项目 | 时长 | 费用 |
|------|------|------|
| 实例开机+环境准备 | 0.5h | 0.75元 |
| 数据上传+模型下载 | 0.5h | 0.75元 |
| 训练（700条，3 epoch） | 1-2h | 1.5-3元 |
| 评测 | 0.5h | 0.75元 |
| **合计** | ~3h | **4-6元** |

预算 100 元，富余很多。如果中途调参再训 1-2 次，最多花 15 元。
