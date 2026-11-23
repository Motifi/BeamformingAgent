# D2D 连续波束赋形

## 图增强数值赋形实验

四个 D2D 收发对各对应一个数值 token。模型用复数 CSI 构建有向干扰图，默认加入正则化参考波束下的接收功率、干扰和速率边际代价。图消息传递和轻量 Transformer 为可微求解层预测交叉干扰权重与正则化系数。求解层读取原始 CSI，输出固定逐 Tx 功率的连续复数波束。

```bash
python -m pip install numpy "torch>=2.4"
python -m d2d_beamforming.benchmark --output graph-benchmark.json
```

三种学习模型使用同一个求解层，以三个随机种子训练，在同一组留出 CSI 上对比图编码、Transformer 和图增强 Transformer。训练直接最大化和速率，模型和正则化基线的参数均通过验证集选择。报告还包括 MRT、固定和验证集调参后的正则化赋形，以及从调参基线出发做 10 步投影梯度的对照。用 `--raw-tokens --test-seed 27002` 可复现只使用复数 CSI 和边能量的首轮实验。这组实验使用合成 CSI，暂不使用预训练骨干。

## 多起点 WMMSE 参考

```bash
python -m d2d_beamforming.wmmse --steps 600 --output wmmse-benchmark.json
```

默认使用相同的 1,024 个留出场景，从 MRT、正则化赋形和四个随机初值求解。每次发射更新在固定功率球面上求解二次子问题。报告记录各起点的收敛速率、逐场景最好的和速率及切空间梯度范数。逐场景多起点最优值用作经验参考。

两条命令均支持 `--links`，每个 Tx 始终使用四个端口。对每个负载分别运行训练和 WMMSE，并保持 `--test-seed`、`--test` 相同。WMMSE 的 `--regularization` 可设为训练报告中的 `tuned_regularization`，使其从该负载验证集选出的正则化波束起步。

## 冻结 DeepSeek 的对照

### DeepSeek 候选求解器

候选头读取冻结 DeepSeek 的链路隐藏状态，并围绕数值波束生成多个满足固定功率的复数候选。按 CSI 上的实际和速率选择候选，可再做有限步 WMMSE 细化。

```bash
python -m d2d_beamforming.llm_solver --links 8 --model /path/to/checkpoint \
  --local-files-only --output warm-start.json --checkpoint beam-solver.pt
python -m d2d_beamforming.llm_infer --checkpoint beam-solver.pt --local-files-only --refine 5
```

推理默认运行该负载的合成示例。可传 `--csi scene.zip --pairs pairs.json --power-dbm 20 --noise-dbm=-110` 读取 SionnaRTStudio 快照，配对数须匹配检查点。

### 接入图增强数值赋形

```bash
python -m d2d_beamforming.llm_benchmark --model /path/to/checkpoint \
  --local-files-only --links 4 6 8 --output graph-deepseek.json
```

每个负载单独训练数值基线，再冻结数值模型和 DeepSeek。适配层把 DeepSeek 隐藏特征映射为链路 token 的残差，通过最终和速率训练。对照组使用同规格适配层读取 DeepSeek 前的投影特征。训练适配层和选择检查点使用独立的合成 CSI，测试与该负载的 WMMSE 共用测试种子和场景数。

### 直接回归复数波束的早期实验

四对单端口 Rx 与四端口 Tx 的全链路复数 CSI，按 `[batch, Rx, Tx, port]` 输入。每个 D2D 配对形成一个链路 token，包含该配对的入边、出边、复数 CSI 和有向边信噪比特征。DeepSeek 的 Transformer 接收四个数值嵌入，四个 token 之间可互相注意。输出头把隐藏状态映射为每个 Tx 的四个复数权重，并逐 Tx 归一化到指定功率。

在仓库根目录运行。默认使用本地加载的 DeepSeek-Coder 1.3B Instruct 权重，首次运行需要下载约 2.69 GB。

```bash
python -m pip install numpy "torch>=2.4" "transformers==4.51.3"
python -m d2d_beamforming --output result.json
python -m d2d_beamforming --model /path/to/checkpoint --local-files-only
```

数值投影和 DeepSeek 权重均冻结。输出头在合成 CSI 上通过最大化和速率训练。报告给出留出 CSI 的平均和速率，以及一个场景的四组复数权重、功率和各方法的和速率。对照包括同规格输出头直接读取投影后的 token、最大比传输（MRT）与考虑发射侧外泄干扰的正则化线性赋形。可用 `--backbone tiny` 运行随机初始化的小型 Transformer 做流程检查。

可附加一个 SionnaRTStudio 导出的场景。设备配对格式见 `d2d_beam_selector/README.md`。示例中的功率和噪声需按实际子载波配置替换。导出场景用于评估，训练仍使用合成 CSI。

```bash
python -m d2d_beamforming --csi scene.zip --pairs pairs.json \
  --power-dbm 20 --noise-dbm=-110 --output result.json
```
