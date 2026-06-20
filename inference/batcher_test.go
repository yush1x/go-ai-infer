package inference

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type recordingPredictor struct {
	mu         sync.Mutex
	batchSizes []int
	err        error
}

func (p *recordingPredictor) Predict(_ context.Context, features []Features) ([]Evaluation, error) {
	p.mu.Lock()
	p.batchSizes = append(p.batchSizes, len(features))
	p.mu.Unlock()

	if p.err != nil {
		return nil, p.err
	}

	results := make([]Evaluation, len(features))
	for i := range features {
		results[i].Value = features[i][0]
	}
	return results, nil
}

func (p *recordingPredictor) sizes() []int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return append([]int(nil), p.batchSizes...)
}

func TestBatcherSendsFullBatch(t *testing.T) {
	predictor := &recordingPredictor{}
	batcher := newTestBatcher(t, predictor, BatcherConfig{
		BatchSize: 3,
		MaxWait:   time.Second,
		QueueSize: 8,
	})

	// 1. 三个并发调用分别代表三盘等待推理的棋局。
	var wg sync.WaitGroup
	errs := make(chan error, 3)
	for i := 1; i <= 3; i++ {
		wg.Add(1)
		go func(value float32) {
			defer wg.Done()

			var features Features
			features[0] = value
			result, err := batcher.Evaluate(context.Background(), features)
			if err == nil && result.Value != value {
				err = errors.New("result was delivered to the wrong request")
			}
			errs <- err
		}(float32(i))
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("Evaluate() error = %v", err)
		}
	}

	if got := predictor.sizes(); len(got) != 1 || got[0] != 3 {
		t.Fatalf("batch sizes = %v, want [3]", got)
	}
}

func TestBatcherSendsPartialBatchAfterMaxWait(t *testing.T) {
	predictor := &recordingPredictor{}
	batcher := newTestBatcher(t, predictor, BatcherConfig{
		BatchSize: 4,
		MaxWait:   10 * time.Millisecond,
		QueueSize: 4,
	})

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if _, err := batcher.Evaluate(ctx, Features{}); err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if got := predictor.sizes(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("batch sizes = %v, want [1]", got)
	}
}

func TestBatcherReturnsPredictorErrorToWholeBatch(t *testing.T) {
	predictErr := errors.New("python unavailable")
	predictor := &recordingPredictor{err: predictErr}
	batcher := newTestBatcher(t, predictor, BatcherConfig{
		BatchSize: 2,
		MaxWait:   time.Second,
		QueueSize: 4,
	})

	errs := make(chan error, 2)
	for range 2 {
		go func() {
			_, err := batcher.Evaluate(context.Background(), Features{})
			errs <- err
		}()
	}

	for range 2 {
		err := <-errs
		if !errors.Is(err, predictErr) {
			t.Fatalf("Evaluate() error = %v, want wrapped predictor error", err)
		}
	}
}

func TestBatcherReportsCompletedBatch(t *testing.T) {
	predictor := &recordingPredictor{}
	batcher := newTestBatcher(t, predictor, BatcherConfig{
		BatchSize: 1,
		MaxWait:   time.Second,
		QueueSize: 1,
	})

	events := make(chan BatchEvent, 1)
	batcher.SetObserver(func(event BatchEvent) {
		events <- event
	})

	if _, err := batcher.Evaluate(context.Background(), Features{}); err != nil {
		t.Fatal(err)
	}
	event := <-events
	if event.Size != 1 || event.Capacity != 1 || event.Duration <= 0 || event.Err != nil {
		t.Fatalf("unexpected batch event: %+v", event)
	}
}

func TestBatcherEvaluateHonorsContextCancellation(t *testing.T) {
	predictor := &recordingPredictor{}
	batcher := newTestBatcher(t, predictor, BatcherConfig{
		BatchSize: 4,
		MaxWait:   time.Second,
		QueueSize: 4,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err := batcher.Evaluate(ctx, Features{})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Evaluate() error = %v, want context deadline exceeded", err)
	}
}

func TestBatcherRejectsCallsAfterClose(t *testing.T) {
	predictor := &recordingPredictor{}
	batcher, err := NewBatcher(predictor, DefaultBatcherConfig())
	if err != nil {
		t.Fatalf("NewBatcher() error = %v", err)
	}
	if err := batcher.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	_, err = batcher.Evaluate(context.Background(), Features{})
	if !errors.Is(err, ErrClosed) {
		t.Fatalf("Evaluate() error = %v, want ErrClosed", err)
	}
}

func newTestBatcher(t *testing.T, predictor BatchPredictor, config BatcherConfig) *Batcher {
	t.Helper()

	batcher, err := NewBatcher(predictor, config)
	if err != nil {
		t.Fatalf("NewBatcher() error = %v", err)
	}
	t.Cleanup(func() {
		if err := batcher.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	return batcher
}
