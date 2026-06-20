package selfplay

import (
	"context"
	"errors"
	"testing"

	"go-ai-infer/board"
	"go-ai-infer/mcts"
)

type scriptedSearcher struct {
	actions []int
	calls   int
	err     error
}

func (s *scriptedSearcher) Search(_ context.Context, _ *board.Board) (*mcts.SearchResult, error) {
	if s.err != nil {
		return nil, s.err
	}
	action := s.actions[s.calls]
	s.calls++
	var policy [mcts.PolicySize]float32
	if action >= 0 && action < len(policy) {
		policy[action] = 1
	}
	return &mcts.SearchResult{Action: action, VisitProbs: policy}, nil
}

func TestPlayReturnsTrainingSamples(t *testing.T) {
	searcher := &scriptedSearcher{actions: []int{0, 1, mcts.PassAction, mcts.PassAction}}
	result := Play(context.Background(), searcher)

	if result.Status != StatusCompleted {
		t.Fatalf("status=%s err=%v, want completed", result.Status, result.Err)
	}
	if result.Game == nil {
		t.Fatal("completed result has nil game")
	}
	if got := len(result.Game.Samples); got != 4 {
		t.Fatalf("samples=%d, want 4", got)
	}

	first := result.Game.Samples[0]
	if first.Player != board.Black || first.Action != 0 {
		t.Errorf("first sample player/action=(%d,%d), want black/0", first.Player, first.Action)
	}
	if first.Features[8*board.Points] != 1 {
		t.Error("first feature should mark black as current player")
	}
	if first.Policy[0] != 1 {
		t.Error("first policy was not copied from search result")
	}

	second := result.Game.Samples[1]
	if second.Player != board.White {
		t.Errorf("second sample player=%d, want white", second.Player)
	}
	if second.Features[8*board.Points] != 0 {
		t.Error("second feature should mark white as current player")
	}
	if first.Score != -second.Score || first.Value != -second.Value {
		t.Errorf("black/white labels should be opposite: first=(%v,%v) second=(%v,%v)",
			first.Value, first.Score, second.Value, second.Score)
	}
	for i, sample := range result.Game.Samples {
		if sample.Ownership != result.Game.Final.Ownership {
			t.Errorf("sample %d ownership does not match final result", i)
		}
	}
}

type maxMovesSearcher struct {
	calls int
}

func (s *maxMovesSearcher) Search(_ context.Context, b *board.Board) (*mcts.SearchResult, error) {
	s.calls++
	action := mcts.PassAction
	if s.calls%2 == 0 {
		mask := b.Mask()
		action = -1
		for p, legal := range mask {
			if legal == 1 {
				action = p
				break
			}
		}
		if action == -1 {
			return nil, errors.New("no legal non-pass action")
		}
	}
	var policy [mcts.PolicySize]float32
	policy[action] = 1
	return &mcts.SearchResult{Action: action, VisitProbs: policy}, nil
}

func TestPlayDiscardsGameAtMaxMoves(t *testing.T) {
	result := Play(context.Background(), &maxMovesSearcher{})
	if result.Status != StatusMaxMoves {
		t.Fatalf("status=%s err=%v, want max_moves", result.Status, result.Err)
	}
	if result.Game != nil {
		t.Fatal("max-moves result must not expose training samples")
	}
	if result.Moves != MaxMoves {
		t.Errorf("moves=%d, want %d", result.Moves, MaxMoves)
	}
	if result.Err == nil {
		t.Error("max-moves result should contain a diagnostic error")
	}
}

func TestPlayUsesConfiguredMaxMovesAndReportsProgress(t *testing.T) {
	var moves []int
	result := PlayWithConfig(context.Background(), &maxMovesSearcher{}, PlayConfig{
		MaxMoves: 3,
		OnMove: func(move int) {
			moves = append(moves, move)
		},
	})

	if result.Status != StatusMaxMoves || result.Moves != 3 {
		t.Fatalf("status=%s moves=%d, want max_moves at 3", result.Status, result.Moves)
	}
	if len(moves) != 3 || moves[0] != 1 || moves[1] != 2 || moves[2] != 3 {
		t.Fatalf("progress=%v, want [1 2 3]", moves)
	}
}

func TestPlayRejectsIllegalAction(t *testing.T) {
	searcher := &scriptedSearcher{actions: []int{-1}}
	result := Play(context.Background(), searcher)
	if result.Status != StatusIllegalAction {
		t.Fatalf("status=%s, want illegal_action", result.Status)
	}
	if result.Game != nil {
		t.Fatal("failed result must not expose training samples")
	}
}

func TestPlayReportsSearchFailure(t *testing.T) {
	searchErr := errors.New("predict failed")
	result := Play(context.Background(), &scriptedSearcher{err: searchErr})
	if result.Status != StatusSearchFailed {
		t.Fatalf("status=%s, want search_failed", result.Status)
	}
	if !errors.Is(result.Err, searchErr) {
		t.Errorf("error=%v, want wrapped search error", result.Err)
	}
}
