# D2D 联合波束选择

从 SionnaRTStudio 导出的完整复数 CFR 计算期望链路与交叉链路增益。兼容 Chat Completions 的模型为每个发射设备选一个波束编号。本地根据同一个码本还原复数波束并重算 WSR，同时输出逐链路最强波束和穷举最优两组对照。

## 运行

在 BeamformingAgent 工作区根目录使用 Python 3.10+。

```bash
python -m pip install numpy
python -m d2d_beam_selector --demo --offline
```

离线示例用固定的数值样本，包含 4 对设备、4 个发射端口和 4 个候选波束。它输出本地结果，省略 API 请求。每个快照的穷举对照最多支持 1,000,000 种组合。

## 配置 LLM

程序从工作区根目录自动读取 `d2d_provider.json`。文件可以配置兼容 OpenAI Chat Completions 的服务地址、模型名和密钥来源。下面是环境变量认证的例子。

```json
{
  "base_url": "https://your-provider.example/v1",
  "model": "your-model-id",
  "api_key_env": "YOUR_PROVIDER_API_KEY"
}
```

当前项目配置使用 DeepSeek 官方端点和 `deepseek-flash`，并在请求中关闭思考模式。使用默认的环境变量认证时，程序读取 `D2D_API_KEY`，调用 DeepSeek 官方端点时也可读取 `DEEPSEEK_API_KEY`。

```bash
python -m d2d_beam_selector --check-config
python -m d2d_beam_selector --demo --output model-demo.json
```

`--check-config` 显示实际请求地址、模型和认证是否就绪。它省略密钥与 URL 查询参数，并且只检查本地配置。运行 `--demo` 或 `--csi` 的在线模式会由 Python 进程向配置的端点发 HTTP 请求。

- 用 `api_url` 可以指定完整的 Chat Completions URL。使用 `--config PATH` 可以选择另一份配置。
- 可以把 `api_key_env` 换成 `api_key_file`，指向一个只含密钥的独立文件。相对路径以配置文件目录为准。macOS 和 Linux 上要求密钥文件对其他用户不可读，推荐权限为 `600`。
- 免认证端点可设置 `"no_auth": true`。不接受 `response_format` 的兼容端点可设置 `"prompt_json": true`。
- 可通过 `extra_body` 配置服务专有参数，如 DeepSeek 的 `thinking`，或以 `max_completion_tokens` 替代默认的 `max_tokens`。
- 端点和模型的优先级依次为命令行、`D2D_API_BASE_URL` 或 `D2D_API_MODEL`、项目配置、内置默认值。临时改用另一域名时，可显式传入对应的密钥环境变量。

```bash
python -m d2d_beam_selector --demo \
  --base-url https://another-provider.example/v1 \
  --model another-model-id \
  --api-key-env ANOTHER_PROVIDER_API_KEY
```

API 请求传递增益数值、发射功率、接收噪声和 WSR 权重。复数 CSI、设备 ID 和原始导出文件保留在本地。响应必须含有效的 `beam_ids`，结果报告会区分模型选择、逐链路选择和穷举最优。

## 接入 SionnaRTStudio

1. 在静态 CSI 仿真中选择所有 D2D 端点，导出 **CIR + CFR** ZIP，也可以直接传其中的 `csi.npz`。
2. 编写 `pairs.json`，用导出中真实的设备 ID 指定 Tx 与 Rx 的配对顺序。每个接收端需为单端口，同一 Tx 侧使用同一阵列配置。

```json
[
  {"tx_id":"tx-1","rx_id":"rx-1"},
  {"tx_id":"tx-2","rx_id":"rx-2"},
  {"tx_id":"tx-3","rx_id":"rx-3"},
  {"tx_id":"tx-4","rx_id":"rx-4"}
]
```

3. 确定每个发射机在这个子载波上的固定功率，以及接收噪声功率。下面是采用 20 dBm 和 −122.2 dBm 的示例，按实验配置替换。可以先加 `--offline` 检查导出和配对。

```bash
python -m d2d_beam_selector \
  --csi /path/to/export.zip --pairs pairs.json \
  --power-dbm 20 --noise-dbm=-122.2 \
  --output d2d-result.json
```

默认读取首个时间样本和最接近 0 Hz 的基带子载波。可用 `--time-index`、`--subcarrier-index` 改选。码本大小由 `--beams` 指定。默认各链路权重相同，也可按配对顺序传 `--weights 1 1 2 2`。

报告包含各候选方法的逐链路频谱效率、WSR、波束编号及各发射端复数波束的实部和虚部。模型结果还包含相对穷举最优的 WSR 差距、延迟和供应商返回的 token 用量。功率在本地通过单位范数码本和每个 Tx 的固定功率构造，评分时考虑所有交叉干扰。
