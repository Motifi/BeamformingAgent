# SWAN-ISAC 数值 CSI 前向模型

按 Gao 等人在 TWC 的摘要及 arXiv 预印本 §III 独立实现。从复数 CSI 构建用户与目标的信道相关图，将图特征注入天线 token，再由 Transformer 和两个输出头给出天线位置、收发段分数及波束权重。代码包只含前向模型和运行入口。

[IEEE TWC](https://doi.org/10.1109/TWC.2026.3749595) · [arXiv §III](https://arxiv.org/html/2604.10372v1#S3)

## 运行

在本目录下使用 Python 3.10 或更新版本。

```bash
python -m pip install "torch>=2.4"
python forward.py
```

默认用随机张量检查轻量 Transformer 的接口。要运行 GPT-2 骨干，安装 Transformers 后选择 `gpt2-pretrained` 或 `gpt2-random`。

```bash
python -m pip install "transformers>=4.51,<5" safetensors
python forward.py --backbone gpt2-pretrained
python forward.py --backbone gpt2-pretrained --gpt2 /path/to/gpt2 --local-files-only
```

`--gpt2` 是标准 GPT-2 small 的模型目录或 ID。LoRA 加在注意力的 Q/K/V 投影上，秩 32，缩放参数 16，dropout 0.05。GPT-2 预训练权重只初始化骨干。需要用 `--weights` 加载与模型结构匹配的任务权重，才能得到经过任务训练的输出。作者的任务权重未找到公开下载。

## 接入自己的 CSI

```python
import torch
from model import SwanModel

model = SwanModel(antennas=40, segments=4).eval()
H_comm = torch.randn(8, 2, 40, dtype=torch.complex64)
H_sense = torch.randn(8, 1, 40, dtype=torch.complex64)
with torch.inference_mode():
    result = model(H_comm, H_sense)
```

`B` 为批大小，`N` 为天线数，`U` 为通信用户数，`S` 为感知目标数，`M` 为分段数。两个输入都采用复数张量，天线轴按波导位置排序。同一模型可处理不同批次的用户数和目标数变化。每个批次内的数量须一致。

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `H_comm` | `[B, U, N]` | 通信 CSI |
| `H_sense` | `[B, S, N]` | 感知 CSI |
| `result.positions_normalized` | `[B, N]` | 归一化天线位置，乘波导长度得到米 |
| `result.segment_scores` | `[B, M]` | 分段得分，给定发射段数 `K` 后可以取前 `K` 名 |
| `result.communication_weights` | `[B, N, U]` | 复数通信波束权重 |
| `result.sensing_weights` | `[B, N, S]` | 复数感知波束权重 |

位置头采用每根天线在其归一化区间内回归的实现，保留天线次序。波束头输出稠密权重，收发掩码、功率投影和自适应 `K` 的目标函数由调用方决定。

也可把两个输入以 `communication_csi` 和 `sensing_csi` 为键保存到一个 `.pt` 字典，用命令行读写。

```bash
python forward.py --input csi.pt --output predictions.pt
```
