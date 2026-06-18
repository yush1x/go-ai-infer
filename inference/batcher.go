package inference

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// Batcher 超参数集中放在文件开头，后续可以直接调整默认值。
const (
	EnableBatchLog   = true // 改为 false 即可关闭所有 batch 日志。
	DefaultBatchSize = 32
	DefaultMaxWait   = 5 * time.Millisecond
	DefaultQueueSize = 128
)

var ErrClosed = errors.New("inference: batcher is closed")

type BatcherConfig struct {
	BatchSize int
	MaxWait   time.Duration
	QueueSize int
}

func DefaultBatcherConfig() BatcherConfig {
	return BatcherConfig{
		BatchSize: DefaultBatchSize,
		MaxWait:   DefaultMaxWait,
		QueueSize: DefaultQueueSize,
	}
}

type evaluateRequest struct {
	ctx      context.Context
	features Features
	response chan evaluateResponse // 缓冲为 1，调用方取消后 batcher 仍可安全写回。
}

type evaluateResponse struct {
	evaluation Evaluation
	err        error
}

// Batcher 对外提供同步 Evaluate，对内串行组织 batch 并调用 BatchPredictor。
type Batcher struct {
	config    BatcherConfig
	predictor BatchPredictor
	requests  chan evaluateRequest

	runCtx context.Context
	cancel context.CancelFunc
	done   chan struct{}

	submitMu  sync.RWMutex // 保证 Close 之后不会再有请求成功进入队列。
	closed    atomic.Bool
	closeOnce sync.Once
}

func NewBatcher(predictor BatchPredictor, config BatcherConfig) (*Batcher, error) {
	if predictor == nil {
		return nil, errors.New("inference: predictor is nil")
	}
	if config.BatchSize <= 0 {
		return nil, errors.New("inference: batch size must be positive")
	}
	if config.MaxWait <= 0 {
		return nil, errors.New("inference: max wait must be positive")
	}
	if config.QueueSize <= 0 {
		return nil, errors.New("inference: queue size must be positive")
	}

	runCtx, cancel := context.WithCancel(context.Background())
	b := &Batcher{
		config:    config,
		predictor: predictor,
		requests:  make(chan evaluateRequest, config.QueueSize),
		runCtx:    runCtx,
		cancel:    cancel,
		done:      make(chan struct{}),
	}

	go b.run()
	return b, nil
}

func (b *Batcher) Evaluate(ctx context.Context, features Features) (Evaluation, error) {
	if ctx == nil {
		return Evaluation{}, errors.New("inference: context is nil")
	}

	b.submitMu.RLock()
	if b.closed.Load() {
		b.submitMu.RUnlock()
		return Evaluation{}, ErrClosed
	}

	req := evaluateRequest{
		ctx:      ctx,
		features: features,
		response: make(chan evaluateResponse, 1),
	}

	// 1. 请求进入有限队列；队列满时形成背压，但调用方仍可通过 context 退出。
	select {
	case b.requests <- req:
	case <-ctx.Done():
		b.submitMu.RUnlock()
		return Evaluation{}, ctx.Err()
	case <-b.done:
		b.submitMu.RUnlock()
		return Evaluation{}, ErrClosed
	}
	b.submitMu.RUnlock()

	// 2. 同步等待属于自己的结果，只阻塞当前调用 Evaluate 的 goroutine。
	select {
	case result := <-req.response:
		return result.evaluation, result.err
	case <-ctx.Done():
		return Evaluation{}, ctx.Err()
	case <-b.done:
		return Evaluation{}, ErrClosed
	}
}

// Close 停止接收新请求，取消正在执行的 HTTP 调用，并唤醒等待中的调用方。
func (b *Batcher) Close() error {
	b.closeOnce.Do(func() {
		b.submitMu.Lock()
		b.closed.Store(true)
		b.cancel()
		b.submitMu.Unlock()
	})
	<-b.done
	return nil
}

func (b *Batcher) run() {
	defer close(b.done)
	defer b.failQueuedRequests(ErrClosed)

	for {
		// 1. 空闲时等待第一个有效请求，不使用固定周期轮询。
		first, ok := b.waitForFirstRequest()
		if !ok {
			return
		}

		// 2. 从第一个请求到达时开始计时，满批或超时都会立即发送。
		batch, ok := b.collectBatch(first)
		if !ok {
			b.replyError(batch, ErrClosed)
			return
		}

		// 3. 一次 HTTP 调用处理整个 batch，再按原顺序分发结果。
		b.executeBatch(batch)
	}
}

func (b *Batcher) waitForFirstRequest() (evaluateRequest, bool) {
	for {
		select {
		case <-b.runCtx.Done():
			return evaluateRequest{}, false
		case req := <-b.requests:
			if req.ctx.Err() != nil {
				b.reply(req, Evaluation{}, req.ctx.Err()) // 已取消请求不占用推理 batch。
				continue
			}
			return req, true
		}
	}
}

func (b *Batcher) collectBatch(first evaluateRequest) ([]evaluateRequest, bool) {
	batch := make([]evaluateRequest, 0, b.config.BatchSize)
	batch = append(batch, first)

	timer := time.NewTimer(b.config.MaxWait)
	defer stopTimer(timer)

	for len(batch) < b.config.BatchSize {
		select {
		case <-b.runCtx.Done():
			return batch, false

		case req := <-b.requests:
			// 2.1 调用方已经取消时直接回错，不把无效请求发给 Python。
			if err := req.ctx.Err(); err != nil {
				b.reply(req, Evaluation{}, err)
				continue
			}
			batch = append(batch, req)

		case <-timer.C:
			// 2.2 未达到 BatchSize 时，最多等待 MaxWait 后发送当前小批次。
			return batch, true
		}
	}

	return batch, true
}

func (b *Batcher) executeBatch(batch []evaluateRequest) {
	features := make([]Features, len(batch))
	for i := range batch {
		features[i] = batch[i].features
	}

	startedAt := time.Now()
	if EnableBatchLog {
		log.Printf("send inference batch: time=%s size=%d", startedAt.Format(time.RFC3339Nano), len(batch))
	}

	results, err := b.predictor.Predict(b.runCtx, features)
	if EnableBatchLog {
		log.Printf("inference finished: size=%d duration=%s err=%v", len(batch), time.Since(startedAt), err)
	}
	if err != nil {
		b.replyError(batch, fmt.Errorf("inference: batch predict: %w", err))
		return
	}
	if len(results) != len(batch) {
		b.replyError(batch, fmt.Errorf(
			"inference: predictor returned %d results for batch of %d",
			len(results),
			len(batch),
		))
		return
	}

	for i, req := range batch {
		b.reply(req, results[i], nil)
	}
}

func (b *Batcher) replyError(batch []evaluateRequest, err error) {
	for _, req := range batch {
		b.reply(req, Evaluation{}, err)
	}
}

func (b *Batcher) reply(req evaluateRequest, evaluation Evaluation, err error) {
	req.response <- evaluateResponse{
		evaluation: evaluation,
		err:        err,
	}
}

func (b *Batcher) failQueuedRequests(err error) {
	for {
		select {
		case req := <-b.requests:
			b.reply(req, Evaluation{}, err)
		default:
			return
		}
	}
}

func stopTimer(timer *time.Timer) {
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
}
