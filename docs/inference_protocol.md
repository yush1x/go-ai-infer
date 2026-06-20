# Python 推理服务协议

Go 端通过 `POST /predict`、`POST /predict/a` 或 `POST /predict/b` 批量调用 Python 模型。三个接口的请求和响应协议完全一致，仅 URL 对应的模型权重不同。请求和响应均为 little-endian、连续排列的 `float32` 二进制数据。

## 模型接口

Python 服务启动时会加载两个模型：

```text
POST /predict    使用默认模型
POST /predict/a  使用服务端配置的模型 a
POST /predict/b  使用服务端配置的模型 b
```

模型权重路径在 `src/server.py` 顶部的 `MODEL_WEIGHTS` 中配置，默认模型由同一位置的 `DEFAULT_MODEL` 配置。

## 请求

```http
POST /predict
Content-Type: application/octet-stream
X-Batch-Size: <B>
```

调用指定模型时只需要把 URL 改为：

```http
POST /predict/a
Content-Type: application/octet-stream
X-Batch-Size: <B>
```

或：

```http
POST /predict/b
Content-Type: application/octet-stream
X-Batch-Size: <B>
```

Body 布局为 NCHW：

```text
shape : [B, 9, 19, 19]
bytes : B * 9 * 19 * 19 * 4
```

单个局面的通道：

```text
0,1 : 当前局面的黑棋、白棋
2,3 : 前 1 个局面的黑棋、白棋
4,5 : 前 2 个局面的黑棋、白棋
6,7 : 前 3 个局面的黑棋、白棋
8   : 当前行动方，黑=1，白=0（整张平面取相同值）
```

元素索引：

```text
index = (((batch * 9 + channel) * 19 + row) * 19 + col)
```

## 响应

响应采用 **head-major** 布局，不是逐样本布局：

```text
policy_probs    [B, 362]
value           [B]
score           [B]
ownership_probs [B, 2, 19, 19]
```

各段依次展平后拼接：

```text
[全部 policy][全部 value][全部 score][全部 ownership]
```

每个样本对应 `1086` 个值，响应总长度为：

```text
floats : B * 1086
bytes  : B * 1086 * 4
```

Go 端切片：

```go
policyEnd := batchSize * 362
valueEnd := policyEnd + batchSize
scoreEnd := valueEnd + batchSize
ownershipEnd := scoreEnd + batchSize*2*19*19

policy := out[:policyEnd]
value := out[policyEnd:valueEnd]
score := out[valueEnd:scoreEnd]
ownership := out[scoreEnd:ownershipEnd]
```

第 `i` 个样本的数据：

```go
samplePolicy := policy[i*362 : (i+1)*362]
sampleValue := value[i]
sampleScore := score[i]
sampleOwnership := ownership[i*2*19*19 : (i+1)*2*19*19]
```

## 输出语义

### Policy

`policy_probs` 是对 362 个动作执行 softmax 后的概率：

```text
0..360 : row * 19 + col
361    : pass
```

概率包含非法动作。Go 端应屏蔽非法动作，然后对剩余概率重新归一化。

### Value

`value` 表示当前行动方的预测胜负，范围为 `[-1, 1]`：

```text
正数：当前行动方占优
负数：当前行动方不利
```

### Score

`score` 表示当前行动方视角的预测终局目差：

```text
正数：当前行动方领先
负数：当前行动方落后
```

训练标签由自博弈终局结果生成。

### Ownership

`ownership_probs` 是每个点的黑白归属概率：

```text
channel 0 : 黑
channel 1 : 白
```

通道固定为黑白视角，不随当前行动方交换。两个通道经过 softmax，不包含中立或未知通道。

训练标签由自博弈终局归属生成；未知点不参与 ownership loss。

## 错误处理

模型在服务启动时加载。若权重文件不存在或模型加载失败，服务启动失败；若模型已配置但当前进程尚未加载完成，请求返回 `503`。

以下情况返回 `400`：

```text
X-Batch-Size 缺失、无效或小于等于 0
Body 长度不等于 B * 9 * 19 * 19 * 4
```

以下情况返回 `404`：

```text
请求了未配置的模型接口，例如 /predict/c
```

以下情况返回 `503`：

```text
模型已配置但尚未完成加载
```

Go 端还应检查：

```text
HTTP 状态码为 200
响应长度等于 B * 1086 * 4
```

## 并发与设备

服务允许并发调用 `/predict`、`/predict/a` 和 `/predict/b`。每个请求会路由到对应的常驻模型实例执行推理。

两个模型使用 `GoInference` 的设备选择逻辑加载：优先 CUDA，其次 MPS，最后 CPU。当前服务不做多 GPU 分配；如果环境只有一张可用 GPU，两个模型会加载到同一张 GPU。
