package match

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"go-ai-infer/board"
	"go-ai-infer/mcts"
)

type passSearcher struct {
	mu    sync.Mutex
	calls int
}

func (s *passSearcher) Search(context.Context, *board.Board) (*mcts.SearchResult, error) {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()
	var policy [mcts.PolicySize]float32
	policy[mcts.PassAction] = 1
	return &mcts.SearchResult{Action: mcts.PassAction, VisitProbs: policy}, nil
}

func TestRunnerAlternatesColorsAndBuildsReport(t *testing.T) {
	a := &passSearcher{}
	b := &passSearcher{}
	runner, err := New(
		Player{Name: "model-a", Searcher: a},
		Player{Name: "model-b", Searcher: b},
		Config{Games: 4, Concurrency: 2, MaxMoves: 10},
	)
	if err != nil {
		t.Fatal(err)
	}

	report, stats, err := runner.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.Completed != 4 || stats.ValidGames() != 4 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if report.Games[0].BlackModel != "model-a" || report.Games[1].BlackModel != "model-b" {
		t.Fatalf("colors were not alternated: %+v %+v", report.Games[0], report.Games[1])
	}
	for _, game := range report.Games {
		if len(game.Frames) != 3 {
			t.Fatalf("game %d frames=%d, want start + two passes", game.ID, len(game.Frames))
		}
	}
	if stats.ModelAWins != 2 || stats.ModelBWins != 2 {
		t.Fatalf("wins should follow white because of komi: %+v", stats)
	}
	if a.calls != 4 || b.calls != 4 {
		t.Fatalf("search calls a=%d b=%d, want 4 each", a.calls, b.calls)
	}
}

func TestWriteReportUsesSingleJSONFile(t *testing.T) {
	report := Report{
		Version: ReportVersion,
		Models:  []string{"a", "b"},
		Games:   []Game{{ID: 1, Status: "completed"}, {ID: 2, Status: "completed"}},
	}
	path, err := WriteReport(filepath.Join(t.TempDir(), "all-games.json"), report)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Report
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Games) != 2 {
		t.Fatalf("games=%d, want 2", len(decoded.Games))
	}
}
