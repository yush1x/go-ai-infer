package inference

import "context"

const (
	BoardSize       = 19
	BoardPoints     = BoardSize * BoardSize
	FeaturePlanes   = 9
	FeatureSize     = FeaturePlanes * BoardPoints
	PolicySize      = BoardPoints + 1
	OwnershipPlanes = 2
	OwnershipSize   = OwnershipPlanes * BoardPoints
	OutputSize      = PolicySize + 1 + 1 + OwnershipSize
)

// Features 是单个局面的连续 NCHW 输入，索引顺序为 channel -> row -> col。
type Features [FeatureSize]float32

// Evaluation 是 Python 模型对单个局面的完整推理结果。
type Evaluation struct {
	Policy    [PolicySize]float32
	Value     float32
	Score     float32
	Ownership [OwnershipSize]float32
}

// Evaluator 是 MCTS 唯一需要依赖的推理接口。
// 调用表现为同步等待，但只阻塞当前 goroutine。
type Evaluator interface {
	Evaluate(ctx context.Context, features Features) (Evaluation, error)
}

// BatchPredictor 负责一次性推理一批局面，Batcher 用它隔离 HTTP 实现。
type BatchPredictor interface {
	Predict(ctx context.Context, features []Features) ([]Evaluation, error)
}
