# SelfPlay 训练数据提交协议

Go 向 Python 保存服务提交一盘正常结束的自博弈数据。HDF5 的分片、压缩、文件名和追加方式由 Python 端决定。

## 请求

```http
POST /selfplay/game
Content-Type: application/octet-stream
X-Sample-Count: <N>
```

- 每个请求提交一盘棋，`N` 为总手数，范围 `1..400`。
- 每个请求提交一盘棋，`N` 为总样本数，必须大于 `0`。
- 所有 `float32` 使用 IEEE 754 little-endian。

## Body

各段按以下顺序连续拼接：

```text
[features][policy][value][score][ownership]

features   float32 [N, 9, 19, 19]
policy     float32 [N, 362]
value      float32 [N]
score      float32 [N]
ownership  int8    [N, 19, 19]
```

单个样本为 `14813` 字节：

```text
features   3249 * 4 = 12996
policy      362 * 4 = 1448
value         1 * 4 = 4
score         1 * 4 = 4
ownership   361 * 1 = 361
```

Body 长度必须严格等于 `N * 14813`。

## 数据语义

Features 表示落子前局面：

```text
0,1 : 当前局面的黑棋、白棋
2,3 : 前 1 手局面的黑棋、白棋
4,5 : 前 2 手局面的黑棋、白棋
6,7 : 前 3 手局面的黑棋、白棋
8   : 当前行动方，黑方全 1，白方全 0
```

Policy 是 MCTS 根节点访问次数分布：

```text
0..360 : row * 19 + col
361    : pass
```

Value 和 Score 使用当前行动方视角：

```text
value : [-1,1] 内的连续值；正数占优，负数不利
score : 正数为领先，负数为落后
```

当前 Value 标签可由终局胜负和该步 MCTS 根节点估值加权生成：

```text
value = (1 - value_mcts_weight) * terminal_value
      + value_mcts_weight * mcts_root_value
```

`terminal_value` 为胜 `1`、负 `-1`、和 `0`。该融合不改变二进制存储格式。

固定贴目为 `7.5`。先计算：

```text
black_lead = black_score - white_score - 7.5
```

黑方行动样本的 Score 为 `black_lead`，白方行动样本取相反数。

Ownership 固定使用黑白视角，不随行动方交换：

```text
0=黑方归属，1=白方归属，-1=中立或未知
```

同一盘棋的所有样本使用相同的最终 Ownership。

## Python 解码参考

```python
import numpy as np

def decode_game(body: bytes, n: int):
    if len(body) != n * 14813:
        raise ValueError("invalid body size")
    offset = 0

    def read(dtype, count):
        nonlocal offset
        size = np.dtype(dtype).itemsize * count
        out = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
        offset += size
        return out

    features = read("<f4", n * 3249).reshape(n, 9, 19, 19)
    policy = read("<f4", n * 362).reshape(n, 362)
    value = read("<f4", n)
    score = read("<f4", n)
    ownership = read("i1", n * 361).reshape(n, 19, 19)
    return features, policy, value, score, ownership
```

## 响应

成功写入返回 HTTP `200`：

```json
{"samples":236,"status":"written"}
```

错误响应格式：

```json
{"error":"invalid_body_size","message":"got 1000, expected 3495868"}
```

状态码：`400` 表示请求非法，`500` 表示保存失败。

Python 端至少校验 `X-Sample-Count`、Body 长度，以及浮点数据中是否存在 NaN 或无穷值。
