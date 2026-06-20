# Inference 使用说明

`inference` 包负责收集多个局面的推理请求，组成 batch 后调用 Python 推理服务。

调用方只需要使用同步的 `Evaluate` 方法，不需要处理 channel、batch 或 HTTP 协议。

## 创建推理器

同一个模型接口应只创建一个共享的 `Batcher`，所有使用该模型的 MCTS 和棋局共同使用它。
比较两个模型时，`/predict/a` 和 `/predict/b` 必须分别创建独立的 `Batcher`，
避免两个模型的局面进入同一个推理 batch。

```go
package main

import (
	"context"
	"log"

	"go-ai-infer/inference"
)

func main() {
	// 1. 创建 Python HTTP 推理客户端。
	client, err := inference.NewHTTPClient(
		inference.DefaultPredictURL,
		inference.DefaultHTTPTimeout,
	)
	if err != nil {
		log.Fatal(err)
	}

	// 2. 创建共享 Batcher。
	batcher, err := inference.NewBatcher(
		client,
		inference.DefaultBatcherConfig(),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer batcher.Close()

	// 3. 推理单个局面。
	var features inference.Features
	result, err := batcher.Evaluate(context.Background(), features)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("value=%f score=%f", result.Value, result.Score)
}
```

默认 Python 接口地址为：

```text
http://127.0.0.1:8000/predict
```

使用其他地址：

```go
client, err := inference.NewHTTPClient(
	"http://192.168.1.10:8000/predict",
	inference.DefaultHTTPTimeout,
)
```

## MCTS 中调用

MCTS 只需要依赖 `inference.Evaluator` 接口：

```go
type MCTS struct {
	evaluator inference.Evaluator
}

func NewMCTS(evaluator inference.Evaluator) *MCTS {
	return &MCTS{
		evaluator: evaluator,
	}
}
```

搜索到叶节点后调用：

```go
result, err := m.evaluator.Evaluate(ctx, board.Features())
if err != nil {
	return err
}

policy := result.Policy
value := result.Value
```

`Evaluate` 会等待当前请求完成，但只阻塞当前 goroutine。其他棋局可以继续提交请求，并由共享的 `Batcher` 自动组成 batch。

## 输入

输入类型：

```go
type Features [9 * 19 * 19]float32
```

数据使用连续 NCHW 布局：

```text
channel -> row -> col
```

元素索引：

```go
index := (channel*inference.BoardSize+row)*inference.BoardSize + col
features[index] = 1
```

通道定义：

```text
0,1 : 当前局面的黑棋、白棋
2,3 : 前 1 个局面的黑棋、白棋
4,5 : 前 2 个局面的黑棋、白棋
6,7 : 前 3 个局面的黑棋、白棋
8   : 当前行动方，黑=1，白=0
```

## 输出

`Evaluate` 返回：

```go
type Evaluation struct {
	Policy    [362]float32
	Value     float32
	Score     float32
	Ownership [722]float32
}
```

- `Policy`：362 个动作的概率，`0..360` 为棋盘点，`361` 为 pass
- `Value`：当前行动方视角的胜负预测
- `Score`：当前行动方视角的预测目差
- `Ownership`：固定黑白视角的点归属概率

`Policy` 包含非法动作。MCTS 必须结合 Board 屏蔽非法动作，然后重新归一化概率。

## Batch 配置

默认配置位于 `batcher.go` 文件开头：

```go
DefaultBatchSize = 32
DefaultMaxWait   = 5 * time.Millisecond
DefaultQueueSize = 128
```

也可以为当前程序单独传入配置：

```go
config := inference.BatcherConfig{
	BatchSize: 64,
	MaxWait:   10 * time.Millisecond,
	QueueSize: 256,
}

batcher, err := inference.NewBatcher(client, config)
```

满足任一条件时会发送请求：

```text
请求数量达到 BatchSize
收到本批第一个请求后等待达到 MaxWait
```

## 日志

默认输出每次 batch 的发送时间、大小、推理耗时和错误：

```text
send inference batch: time=2026-06-18T10:30:01.123+08:00 size=32
inference finished: size=32 duration=6ms err=<nil>
```

关闭日志：修改 `batcher.go` 文件开头的配置：

```go
EnableBatchLog = false
```

## 超时和错误

调用方可以使用 `context` 控制单次等待时间：

```go
ctx, cancel := context.WithTimeout(
	context.Background(),
	30*time.Second,
)
defer cancel()

result, err := batcher.Evaluate(ctx, features)
```

以下情况会返回错误：

- 调用方的 `context` 被取消或超时
- Python 服务无法连接或 HTTP 请求超时
- Python 服务返回非 `200` 状态码
- 响应长度或格式不符合协议
- `Batcher` 已关闭

MCTS 收到错误后应终止本次搜索并向上返回错误，不要使用全零结果继续搜索。

## 关闭

程序退出前调用：

```go
defer batcher.Close()
```

关闭后不再接受新请求，正在等待的调用会收到 `inference.ErrClosed`。
