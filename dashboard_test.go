package main

import (
	"strings"
	"testing"
)

func TestGameLinesAdaptToWidth(t *testing.T) {
	d := &dashboard{
		games: []gameDisplay{
			{status: "running", moves: 10},
			{status: "completed", moves: 20},
			{status: "max_moves", moves: 400},
			{status: "waiting", moves: 0},
			{status: "saving", moves: 30},
		},
	}

	if got := len(d.gameLinesLocked(gameCellWidth)); got != 5 {
		t.Fatalf("narrow lines=%d, want 5", got)
	}
	if got := len(d.gameLinesLocked(3 * gameCellWidth)); got != 2 {
		t.Fatalf("wide lines=%d, want 2", got)
	}
	if line := d.gameLinesLocked(3 * gameCellWidth)[0]; !strings.Contains(line, "#003") {
		t.Fatalf("first wide line=%q, want three games", line)
	}
}

func TestDisplayWidthCountsChineseCharacters(t *testing.T) {
	if got := displayWidth("A已保存"); got != 7 {
		t.Fatalf("display width=%d, want 7", got)
	}
}
