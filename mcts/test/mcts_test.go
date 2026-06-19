package test

import (
	"context"
	"fmt"
	"testing"

	"go-ai-infer/board"
	"go-ai-infer/inference"
	"go-ai-infer/mcts"
)

type mctsEvaluator struct {
	callCount int
}

func (m *mctsEvaluator) Evaluate(_ context.Context, f inference.Features) (inference.Evaluation, error) {
	m.callCount++
	eval := inference.Evaluation{Value: 0.1}
	for p := 0; p < inference.BoardPoints; p++ {
		eval.Policy[p] = 1.0 / float32(inference.PolicySize)
	}
	eval.Policy[inference.BoardPoints] = 1.0 / float32(inference.PolicySize)
	return eval, nil
}

func TestMCTS_EmptyBoard(t *testing.T) {
	fmt.Println("=== Test 7: MCTS empty board ===")
	eval := &mctsEvaluator{}
	cfg := mcts.DefaultConfig()
	cfg.NumSimulations = 100
	cfg.SelfPlay = true
	searcher := mcts.NewSearcher(eval, cfg)
	b := board.New()

	result, err := searcher.Search(context.Background(), b)
	if err != nil {
		t.Fatalf("Search fail: %v", err)
	}
	if result.Action < 0 || result.Action > 361 {
		t.Errorf("invalid action: %d", result.Action)
	}
	x := result.Action / board.Size
	y := result.Action % board.Size
	fmt.Printf("  action=(%d,%d) rootValue=%.4f\n", x, y, result.RootValue)

	var sum float32
	for _, v := range result.VisitProbs {
		sum += v
	}
	fmt.Printf("  visitProbs sum=%.4f CNN calls=%d\n", sum, eval.callCount)
	if sum > 0 && (sum < 0.99 || sum > 1.01) {
		t.Errorf("visitProbs sum should be ~1.0, got %.4f", sum)
	}
}

func TestMCTS_TerminalBoard(t *testing.T) {
	fmt.Println("=== Test 8: MCTS terminal board ===")
	eval := &mctsEvaluator{}
	cfg := mcts.DefaultConfig()
	cfg.NumSimulations = 10
	searcher := mcts.NewSearcher(eval, cfg)
	b := board.New()
	b.Move(-1, -1)
	b.Move(-1, -1)

	if b.IsFinish() != 1 {
		t.Fatal("should be terminal")
	}
	result, err := searcher.Search(context.Background(), b)
	if err != nil {
		t.Fatalf("Search fail: %v", err)
	}
	if result.Action != 361 {
		t.Errorf("terminal should return pass(361), got %d", result.Action)
	}
	fmt.Printf("  action=pass rootValue=%.4f CNN calls=%d\n", result.RootValue, eval.callCount)
	if eval.callCount > 0 {
		t.Error("terminal should NOT call CNN")
	}
}

func TestMCTS_WithMoves(t *testing.T) {
	fmt.Println("=== Test 9: MCTS with moves ===")
	eval := &mctsEvaluator{}
	cfg := mcts.DefaultConfig()
	cfg.NumSimulations = 80
	searcher := mcts.NewSearcher(eval, cfg)

	b := board.New()
	moves := [][2]int{{3, 3}, {15, 15}, {3, 15}, {15, 3}}
	for _, m := range moves {
		if b.Move(m[0], m[1]) != 0 {
			t.Fatalf("move (%d,%d) fail", m[0], m[1])
		}
	}
	fmt.Printf("  round=%d (black to move)\n", b.Round())

	result, err := searcher.Search(context.Background(), b)
	if err != nil {
		t.Fatalf("Search fail: %v", err)
	}
	x := result.Action / board.Size
	y := result.Action % board.Size
	fmt.Printf("  action=(%d,%d) rootValue=%.4f\n", x, y, result.RootValue)

	mask := b.Mask()
	if result.Action != 361 && mask[result.Action] != 1 {
		t.Errorf("chosen action (%d,%d) not legal", x, y)
	}
}

func TestMCTS_SelfPlayMode(t *testing.T) {
	fmt.Println("=== Test 10: MCTS self-play mode ===")
	eval := &mctsEvaluator{}
	cfg := mcts.DefaultConfig()
	cfg.NumSimulations = 100
	cfg.SelfPlay = true
	searcher := mcts.NewSearcher(eval, cfg)
	b := board.New()

	actions := make(map[int]int)
	for i := 0; i < 5; i++ {
		result, err := searcher.Search(context.Background(), b)
		if err != nil {
			t.Fatalf("search %d fail: %v", i+1, err)
		}
		actions[result.Action]++
	}
	fmt.Printf("  5 searches action distribution: %v\n", actions)

	result, _ := searcher.Search(context.Background(), b)
	nonZero := 0
	for _, v := range result.VisitProbs {
		if v > 0 {
			nonZero++
		}
	}
	fmt.Printf("  visitProbs non-zero: %d\n", nonZero)
	if nonZero == 0 {
		t.Error("visitProbs should not be all zero")
	}
}

func TestMCTS_TerminalValue(t *testing.T) {
	fmt.Println("=== Test 11: MCTS terminal value with komi ===")
	eval := &mctsEvaluator{}
	cfg := mcts.DefaultConfig()
	cfg.NumSimulations = 10
	searcher := mcts.NewSearcher(eval, cfg)

	b := board.New()
	for x := 0; x < 10; x++ {
		for y := 0; y < 19; y++ {
			b.Move(x, y)
			b.Move(18-x, 18-y)
		}
	}
	b.Move(-1, -1)
	b.Move(-1, -1)

	if b.IsFinish() != 1 {
		t.Fatal("should be terminal")
	}
	score := b.Score()
	fmt.Printf("  score: black=%d white=%d neutral=%d\n", score[0], score[1], score[2])

	result, err := searcher.Search(context.Background(), b)
	if err != nil {
		t.Fatalf("Search fail: %v", err)
	}
	fmt.Printf("  rootValue=%.4f\n", result.RootValue)
	if result.RootValue <= 0 {
		t.Errorf("black ahead, rootValue should be >0, got %.4f", result.RootValue)
	}
}

func TestMCTS_NoEyeFill(t *testing.T) {
	fmt.Println("=== Test 12: MCTS no eye fill ===")
	eval := &mctsEvaluator{}
	cfg := mcts.DefaultConfig()
	cfg.NumSimulations = 80
	searcher := mcts.NewSearcher(eval, cfg)

	b := board.New()
	eyeMoves := [][2]int{
		{0, 0}, {18, 18}, {0, 1}, {18, 17}, {0, 2}, {18, 16},
		{1, 0}, {18, 15}, {1, 2}, {18, 14}, {2, 0}, {18, 13},
		{2, 1}, {18, 12}, {2, 2}, {18, 11},
	}
	for _, m := range eyeMoves {
		if b.Move(m[0], m[1]) != 0 {
			t.Fatalf("move (%d,%d) fail", m[0], m[1])
		}
	}
	if b.IsEyes(1, 1) != 1 {
		t.Skip("(1,1) is not an eye, skip")
	}
	fmt.Println("  (1,1) is black's eye")

	result, err := searcher.Search(context.Background(), b)
	if err != nil {
		t.Fatalf("Search fail: %v", err)
	}
	if result.Action == 1*board.Size+1 {
		t.Error("MCTS should NOT fill own eye (1,1)")
	} else {
		x := result.Action / board.Size
		y := result.Action % board.Size
		fmt.Printf("  correctly avoided eye, chose (%d,%d)\n", x, y)
	}
}

func TestMCTS_VisitProbsOutput(t *testing.T) {
	fmt.Println("=== Test 13: MCTS visitProbs output ===")
	eval := &mctsEvaluator{}
	cfg := mcts.DefaultConfig()
	cfg.NumSimulations = 200
	searcher := mcts.NewSearcher(eval, cfg)
	b := board.New()

	result, err := searcher.Search(context.Background(), b)
	if err != nil {
		t.Fatalf("Search fail: %v", err)
	}

	var sum float32
	nonZero := 0
	maxProb := float32(0)
	maxAction := -1
	for action, prob := range result.VisitProbs {
		sum += prob
		if prob > 0 {
			nonZero++
		}
		if prob > maxProb {
			maxProb = prob
			maxAction = action
		}
	}

	fmt.Printf("  len=%d sum=%.6f nonZero=%d/%d\n", len(result.VisitProbs), sum, nonZero, len(result.VisitProbs))
	fmt.Printf("  maxProb=%.4f action=%d chosen=%d rootValue=%.4f\n", maxProb, maxAction, result.Action, result.RootValue)

	if sum > 0 && (sum < 0.99 || sum > 1.01) {
		t.Errorf("visitProbs sum should be ~1.0, got %.4f", sum)
	}
	if nonZero == 0 {
		t.Error("visitProbs should not be all zero")
	}
}
