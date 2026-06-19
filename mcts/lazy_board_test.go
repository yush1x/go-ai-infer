package mcts

import (
	"context"
	"testing"

	"go-ai-infer/board"
	"go-ai-infer/inference"
)

type uniformEvaluator struct{}

func (uniformEvaluator) Evaluate(context.Context, inference.Features) (inference.Evaluation, error) {
	var eval inference.Evaluation
	eval.Value = 0.25
	for i := range eval.Policy {
		eval.Policy[i] = 1
	}
	return eval, nil
}

func TestExpandCreatesLazyChildBoards(t *testing.T) {
	searcher := NewSearcher(uniformEvaluator{}, DefaultConfig())
	root := newRootNode(board.New())

	if _, err := searcher.expand(context.Background(), root, false); err != nil {
		t.Fatalf("expand root: %v", err)
	}
	if len(root.children) != PolicySize {
		t.Fatalf("children = %d, want %d", len(root.children), PolicySize)
	}

	for action, child := range root.children {
		if child.board != nil {
			t.Fatalf("child %d eagerly created a board", action)
		}
		if child.parent != root {
			t.Fatalf("child %d has wrong parent", action)
		}
	}
}

func TestEnsureBoardCreatesOnlySelectedChild(t *testing.T) {
	rootBoard := board.New()
	root := newRootNode(rootBoard)
	first := newChildNode(root, 0, 0.5)
	second := newChildNode(root, 1, 0.5)

	if err := first.ensureBoard(); err != nil {
		t.Fatalf("ensure selected child board: %v", err)
	}
	if first.board == nil {
		t.Fatal("selected child board was not created")
	}
	if second.board != nil {
		t.Fatal("unselected child board was created")
	}
	if rootBoard.Round() != 0 {
		t.Fatalf("root board was modified: round=%d", rootBoard.Round())
	}
	if first.board.Round() != 1 {
		t.Fatalf("child board round=%d, want 1", first.board.Round())
	}
	if first.board.Mask()[0] != 0 {
		t.Fatal("selected action was not applied to child board")
	}
}

func TestSimulationMaterializesOnePath(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumSimulations = 1
	searcher := NewSearcher(uniformEvaluator{}, cfg)
	root := newRootNode(board.New())

	if _, err := searcher.expand(context.Background(), root, false); err != nil {
		t.Fatalf("expand root: %v", err)
	}
	if err := searcher.simulate(context.Background(), root, 0); err != nil {
		t.Fatalf("simulate: %v", err)
	}

	materialized := 0
	for _, child := range root.children {
		if child.board != nil {
			materialized++
		}
	}
	if materialized != 1 {
		t.Fatalf("materialized child boards=%d, want 1", materialized)
	}
}
