package runner

import (
	"context"
	"errors"
	"io"
	"log"
	"sync"
	"testing"

	"go-ai-infer/board"
	"go-ai-infer/mcts"
	"go-ai-infer/selfplay"
)

type passSearcher struct{}

func (passSearcher) Search(context.Context, *board.Board) (*mcts.SearchResult, error) {
	var policy [mcts.PolicySize]float32
	policy[mcts.PassAction] = 1
	return &mcts.SearchResult{Action: mcts.PassAction, VisitProbs: policy}, nil
}

type countingSaver struct {
	mu        sync.Mutex
	calls     int
	failCalls map[int]bool
}

func (s *countingSaver) SaveGame(context.Context, *selfplay.Game) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	if s.failCalls[s.calls] {
		return errors.New("save failed")
	}
	return nil
}

func TestRunnerRunsAndSavesGames(t *testing.T) {
	saver := &countingSaver{}
	r, err := New(passSearcher{}, saver, Config{Games: 5, Concurrency: 3})
	if err != nil {
		t.Fatal(err)
	}
	r.SetLogger(log.New(io.Discard, "", 0))

	stats, err := r.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.Started != 5 || stats.Completed != 5 || stats.Saved != 5 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if stats.Samples != 10 {
		t.Errorf("samples=%d, want 10", stats.Samples)
	}
	if saver.calls != 5 {
		t.Errorf("save calls=%d, want 5", saver.calls)
	}
}

func TestRunnerDropsSaveFailureWithoutRetry(t *testing.T) {
	saver := &countingSaver{failCalls: map[int]bool{2: true}}
	r, err := New(passSearcher{}, saver, Config{Games: 3, Concurrency: 2})
	if err != nil {
		t.Fatal(err)
	}
	r.SetLogger(log.New(io.Discard, "", 0))

	stats, err := r.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.Saved != 2 || stats.SaveFailed != 1 {
		t.Fatalf("unexpected save stats: %+v", stats)
	}
	if saver.calls != 3 {
		t.Errorf("save calls=%d, want exactly 3 without retry", saver.calls)
	}
}

type blockingSearcher struct{}

func (blockingSearcher) Search(ctx context.Context, _ *board.Board) (*mcts.SearchResult, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

type nonFinishingSearcher struct {
	calls int
}

func (s *nonFinishingSearcher) Search(_ context.Context, b *board.Board) (*mcts.SearchResult, error) {
	s.calls++
	action := mcts.PassAction
	if s.calls%2 == 0 {
		for point, legal := range b.Mask() {
			if legal == 1 {
				action = point
				break
			}
		}
	}
	var policy [mcts.PolicySize]float32
	policy[action] = 1
	return &mcts.SearchResult{Action: action, VisitProbs: policy}, nil
}

func TestRunnerSavesGameAtMaxMoves(t *testing.T) {
	saver := &countingSaver{}
	r, err := New(&nonFinishingSearcher{}, saver, Config{
		Games: 1, Concurrency: 1, MaxMoves: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	r.SetLogger(log.New(io.Discard, "", 0))

	stats, err := r.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.MaxMoves != 1 || stats.Saved != 1 || stats.Samples != 3 {
		t.Fatalf("unexpected max-moves stats: %+v", stats)
	}
	if saver.calls != 1 {
		t.Fatalf("save calls=%d, want 1", saver.calls)
	}
}

func TestRunnerStopsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	r, err := New(blockingSearcher{}, &countingSaver{}, Config{Games: 10, Concurrency: 2})
	if err != nil {
		t.Fatal(err)
	}
	r.SetLogger(log.New(io.Discard, "", 0))
	cancel()

	stats, err := r.Run(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v, want context canceled", err)
	}
	if stats.Saved != 0 {
		t.Errorf("saved=%d, want 0", stats.Saved)
	}
}

func TestRunnerRejectsInvalidValueMCTSWeight(t *testing.T) {
	_, err := New(passSearcher{}, &countingSaver{}, Config{
		Games:           1,
		Concurrency:     1,
		ValueMCTSWeight: -0.1,
	})
	if err == nil {
		t.Fatal("expected invalid value MCTS weight error")
	}
}
