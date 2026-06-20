package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteTraceCreatesJSON(t *testing.T) {
	trace := gameTrace{
		Status: "completed",
		Moves: []moveTrace{{
			Move:       1,
			Player:     "black",
			Coordinate: "(3,3)",
			Board:      make([]int, 361),
		}},
	}

	path, err := writeTrace(filepath.Join(t.TempDir(), "named-game.json"), trace)
	if err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var decoded gameTrace
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Status != "completed" || len(decoded.Moves) != 1 {
		t.Fatalf("unexpected decoded trace: %+v", decoded)
	}
}
