package main

import (
	"strings"
	"testing"
	"time"

	"go-ai-infer/inference"
)

func TestMatchDashboardSummaryIncludesWinsAndRunningGames(t *testing.T) {
	d := &matchDashboard{
		modelA: "a",
		modelB: "b",
		games: []matchGameDisplay{
			{status: "running", moves: 20, blackModel: "a", whiteModel: "b"},
			{status: "completed", moves: 200, winner: "a"},
			{status: "max_moves", moves: 450, winner: "b"},
			{status: "search_failed", moves: 3},
		},
		startedAt: time.Now(),
	}
	line := d.matchSummaryLocked()
	for _, want := range []string{"3/4", "运行 1", "a胜 1", "b胜 1", "失败 1"} {
		if !strings.Contains(line, want) {
			t.Fatalf("summary=%q, missing %q", line, want)
		}
	}
}

func TestMatchDashboardGameLinesShowEveryGame(t *testing.T) {
	d := &matchDashboard{
		games: []matchGameDisplay{
			{status: "running", moves: 10, blackModel: "a", whiteModel: "b"},
			{status: "completed", moves: 20, blackModel: "b", whiteModel: "a", winner: "a"},
			{status: "waiting"},
		},
	}
	lines := d.gameLinesLocked(2 * matchGameCellWidth)
	joined := strings.Join(lines, "\n")
	for _, want := range []string{"#001", "#002", "#003", "a胜"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("lines=%q, missing %q", joined, want)
		}
	}
}

func TestMatchDashboardTracksBatchesSeparately(t *testing.T) {
	d := &matchDashboard{
		batchA: newMatchBatchDisplay(8),
		batchB: newMatchBatchDisplay(8),
	}
	d.OnBatchA(inference.BatchEvent{Size: 8, Capacity: 8, Duration: 2 * time.Millisecond})
	d.OnBatchB(inference.BatchEvent{Size: 3, Capacity: 8, Duration: 4 * time.Millisecond})

	if d.batchA.count != 1 || d.batchA.full != 1 {
		t.Fatalf("batch A=%+v", d.batchA)
	}
	if d.batchB.count != 1 || d.batchB.full != 0 {
		t.Fatalf("batch B=%+v", d.batchB)
	}
}
